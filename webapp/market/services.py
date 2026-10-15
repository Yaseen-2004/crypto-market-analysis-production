"""The refresh job and the read side of the market data.

``refresh_market_data`` is a plain function so it can be driven by anything:
the APScheduler loop here, a management command, cron, or a Celery task.
"""

from __future__ import annotations

import logging
import threading
from datetime import timedelta

from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils import timezone

from src.config import AnalysisConfig
from src.main import run_pipeline
from src.ui_payload import build_payload

from .models import DataSnapshot, RefreshRun, SchedulerState

logger = logging.getLogger("market")

# Guards against two refreshes in one process (scheduler tick + manual button).
# Across processes, the RUNNING row in the database is the guard.
_process_lock = threading.Lock()


def run_timeout() -> timedelta:
    return timedelta(minutes=settings.CRYPTO_RUN_TIMEOUT_MINUTES)


def active_run() -> RefreshRun | None:
    """A run currently in progress in any process (ignoring ones that died)."""
    return (RefreshRun.objects
            .filter(status=RefreshRun.Status.RUNNING, started_at__gte=timezone.now() - run_timeout())
            .first())


def refresh_market_data(trigger: str = RefreshRun.Trigger.SCHEDULE, client=None,
                        **config_overrides) -> RefreshRun | None:
    """Fetch, analyse and store a fresh snapshot. Returns the run, or None if skipped.

    Never raises: a failure is recorded on the run so the scheduler keeps going
    and the interface can say what went wrong.
    """
    if not _process_lock.acquire(blocking=False):
        logger.info("Refresh skipped: another refresh is already running in this process")
        return None
    try:
        if active_run():
            logger.info("Refresh skipped: another process is already refreshing")
            return None

        run = RefreshRun.objects.create(trigger=trigger)
        logger.info("Refresh #%s started (%s)", run.pk, trigger)
        try:
            options = dict(
                top_n=settings.CRYPTO_TOP_N,
                days=settings.CRYPTO_DAYS,
                cache_minutes=0,                                   # live snapshot: always fresh
                history_cache_minutes=settings.CRYPTO_HISTORY_CACHE_MINUTES,
                make_plots=False,
                make_interactive=False,
                use_database=True,
            )
            options.update(config_overrides)
            result = run_pipeline(AnalysisConfig(**options), client=client)

            finished = timezone.now()
            payload = build_payload(result.summary, result.analysis, result.outlier_report,
                                    refreshed_at=finished, run_id=run.pk)
            log = result.fetch.log
            with transaction.atomic():
                DataSnapshot.objects.create(run=run, payload=payload, created_at=finished)
                run.status = RefreshRun.Status.SUCCESS
                run.finished_at = finished
                run.coins = payload["meta"]["coins"]
                run.observations = payload["meta"]["observations"]
                run.outliers = payload["meta"]["totalOutliers"]
                run.charts_downloaded = int((log["source"] == "api").sum()) if "source" in log else None
                run.window_end = result.analysis["timestamp"].max().to_pydatetime()
                run.save()
            prune_snapshots()
            logger.info("Refresh #%s finished in %.0fs: %s coins, %s outliers, %s histories downloaded",
                        run.pk, run.duration_seconds, run.coins, run.outliers, run.charts_downloaded)
        except Exception as exc:  # a background job must record failure, not crash the scheduler
            logger.exception("Refresh #%s failed", run.pk)
            run.status = RefreshRun.Status.FAILED
            run.finished_at = timezone.now()
            run.error = f"{type(exc).__name__}: {exc}"[:2000]
            run.save()
        return run
    finally:
        _process_lock.release()


def run_job(trigger: str = RefreshRun.Trigger.SCHEDULE) -> None:
    """Entry point for threads and the scheduler: owns its database connections."""
    close_old_connections()
    try:
        refresh_market_data(trigger=trigger)
    finally:
        close_old_connections()


def start_refresh_in_background(trigger: str = RefreshRun.Trigger.MANUAL) -> threading.Thread:
    thread = threading.Thread(target=run_job, kwargs={"trigger": trigger}, name="market-refresh", daemon=True)
    thread.start()
    return thread


def prune_snapshots() -> None:
    """Keep only the newest payloads; run history rows are kept for the log."""
    keep = list(DataSnapshot.objects.values_list("pk", flat=True)[: settings.CRYPTO_SNAPSHOTS_TO_KEEP])
    DataSnapshot.objects.exclude(pk__in=keep).delete()


def mark_interrupted_runs() -> int:
    """Runs left RUNNING by a killed process would otherwise block refreshes until they time out."""
    return (RefreshRun.objects
            .filter(status=RefreshRun.Status.RUNNING, started_at__lt=timezone.now() - run_timeout())
            .update(status=RefreshRun.Status.FAILED, finished_at=timezone.now(),
                    error="Interrupted: the process stopped before the refresh finished."))


def latest_snapshot() -> DataSnapshot | None:
    return DataSnapshot.objects.select_related("run").first()


def last_success() -> RefreshRun | None:
    return RefreshRun.objects.filter(status=RefreshRun.Status.SUCCESS).first()


def _iso(value):
    return value.isoformat(timespec="seconds") if value else None


def status() -> dict:
    """Everything the live indicator needs, in one small response."""
    now = timezone.now()
    success = last_success()
    latest = RefreshRun.objects.first()
    running = active_run()
    state = SchedulerState.objects.filter(pk=1).first()
    alive = bool(state and state.heartbeat_at >= now - timedelta(seconds=90))
    return {
        "serverTime": _iso(now),
        "intervalMinutes": settings.CRYPTO_REFRESH_INTERVAL_MINUTES,
        "running": bool(running),
        "runningSince": _iso(running.started_at) if running else None,
        "scheduler": {
            "alive": alive,
            "nextRunAt": _iso(state.next_run_at) if alive else None,
            "heartbeatAt": _iso(state.heartbeat_at) if state else None,
        },
        "lastSuccess": {
            "id": success.pk,
            "finishedAt": _iso(success.finished_at),
            "durationSeconds": success.duration_seconds,
            "coins": success.coins,
            "outliers": success.outliers,
            "windowEnd": _iso(success.window_end),
        } if success else None,
        "lastRun": {
            "id": latest.pk,
            "status": latest.status,
            "error": latest.error or None,
            "finishedAt": _iso(latest.finished_at),
        } if latest else None,
    }
