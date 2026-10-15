"""The 15-minute refresh loop (APScheduler).

Two ways to run it:

* inside the development server - started automatically by ``manage.py runserver``;
* as its own process - ``manage.py runscheduler``, with CRYPTO_SCHEDULER_AUTOSTART=0
  on the web server so exactly one scheduler exists.

The refresh is anchored to a fixed cadence (T, T+15, T+30 ...). If a refresh
overruns its slot - the hourly history download can, on the free API tier - the
overlapping tick is skipped (max_instances=1) and missed ticks collapse into one
(coalesce), so refreshes never pile up.
"""

from __future__ import annotations

import logging
import os
import socket
from datetime import timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from django.conf import settings
from django.db import OperationalError, ProgrammingError, close_old_connections
from django.utils import timezone

from . import services

logger = logging.getLogger("market")

REFRESH_JOB_ID = "refresh-market-data"
HEARTBEAT_JOB_ID = "scheduler-heartbeat"

_scheduler: BaseScheduler | None = None


def first_run_time(now=None):
    """Refresh straight away if the data is stale, otherwise continue the existing cadence."""
    now = now or timezone.now()
    interval = timedelta(minutes=settings.CRYPTO_REFRESH_INTERVAL_MINUTES)
    success = services.last_success()
    if success is None or success.finished_at is None or now - success.finished_at >= interval:
        return now + timedelta(seconds=3)
    return success.finished_at + interval


def add_jobs(scheduler: BaseScheduler, first_run) -> None:
    minutes = settings.CRYPTO_REFRESH_INTERVAL_MINUTES
    scheduler.add_job(
        services.run_job,
        trigger=IntervalTrigger(minutes=minutes, timezone="UTC"),
        id=REFRESH_JOB_ID,
        name=f"Refresh market data every {minutes} minutes",
        kwargs={"trigger": "schedule"},
        next_run_time=first_run,
        max_instances=1,          # never two refreshes at once
        coalesce=True,            # missed ticks collapse into a single run
        misfire_grace_time=minutes * 60,
        replace_existing=True,
    )
    scheduler.add_job(
        heartbeat, trigger=IntervalTrigger(seconds=30, timezone="UTC"), id=HEARTBEAT_JOB_ID,
        next_run_time=timezone.now(), max_instances=1, coalesce=True, replace_existing=True,
    )


def heartbeat() -> None:
    """Publish liveness and the next refresh time for the web process to read."""
    close_old_connections()
    try:
        from .models import SchedulerState

        job = _scheduler.get_job(REFRESH_JOB_ID) if _scheduler else None
        SchedulerState.objects.update_or_create(pk=1, defaults={
            "heartbeat_at": timezone.now(),
            "next_run_at": job.next_run_time if job else None,
            "interval_minutes": settings.CRYPTO_REFRESH_INTERVAL_MINUTES,
            "process": f"{socket.gethostname()}:{os.getpid()}"[:80],
        })
    except (OperationalError, ProgrammingError):
        logger.warning("Scheduler heartbeat could not be written - run `python manage.py migrate`")
    finally:
        close_old_connections()


def bootstrap(scheduler: BaseScheduler) -> None:
    """Database work that must not happen during app loading."""
    close_old_connections()
    try:
        cleared = services.mark_interrupted_runs()
        if cleared:
            logger.warning("Marked %d interrupted refresh run(s) as failed", cleared)
        first = first_run_time()
    except (OperationalError, ProgrammingError):
        logger.error("Scheduler not started: database tables are missing - run `python manage.py migrate`")
        return
    finally:
        close_old_connections()
    add_jobs(scheduler, first)
    wait = max(0, (first - timezone.now()).total_seconds())
    logger.info("Scheduler running: refresh every %d minutes, next in %.0f s",
                settings.CRYPTO_REFRESH_INTERVAL_MINUTES, wait)


def start(blocking: bool = False) -> BaseScheduler:
    """Start the scheduler once per process."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    scheduler_cls = BlockingScheduler if blocking else BackgroundScheduler
    _scheduler = scheduler_cls(timezone="UTC")
    # Bootstrap runs as the scheduler's first job, so no queries run during app loading.
    _scheduler.add_job(bootstrap, args=[_scheduler], id="bootstrap", next_run_time=timezone.now())
    logger.info("Starting %s refresh scheduler", "blocking" if blocking else "background")
    _scheduler.start()
    return _scheduler


def shutdown() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
