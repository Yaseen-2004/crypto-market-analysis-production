"""Tests for the refresh loop, its API and the served interfaces.

    python webapp/manage.py test market

The refresh tests run the real pipeline against the offline fake API client
from the main test-suite, with every pipeline output redirected to a
temporary directory.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from unittest import mock

from django.conf import settings
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from market import scheduler, services
from market.models import DataSnapshot, RefreshRun, SchedulerState
from tests.helpers import FakeCoinGeckoClient


@contextmanager
def isolated_outputs():
    """Point every pipeline output path at a temporary directory."""
    from src import config

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root, real = Path(tmp), config.PROJECT_ROOT
        originals = {}
        for name in dir(config):
            value = getattr(config, name)
            if name.isupper() and isinstance(value, Path):
                try:
                    relative = value.relative_to(real)
                except ValueError:
                    continue
                originals[name] = value
                setattr(config, name, root / relative)
        try:
            yield root
        finally:
            for name, value in originals.items():
                setattr(config, name, value)


def fake_refresh(**overrides):
    with isolated_outputs():
        return services.refresh_market_data(
            trigger=RefreshRun.Trigger.MANUAL, client=FakeCoinGeckoClient(),
            top_n=6, min_observations=24, **overrides)


def make_success(minutes_ago: int = 0, run_id_payload: bool = True) -> RefreshRun:
    finished = timezone.now() - timedelta(minutes=minutes_ago)
    run = RefreshRun.objects.create(status=RefreshRun.Status.SUCCESS, started_at=finished - timedelta(seconds=20),
                                    finished_at=finished, coins=6, outliers=4)
    DataSnapshot.objects.create(run=run, created_at=finished,
                                payload={"meta": {"runId": run.pk if run_id_payload else None}, "coins": []})
    return run


# ---------------------------------------------------------------------------
# The refresh job
# ---------------------------------------------------------------------------
class RefreshJobTests(TestCase):
    def test_refresh_stores_a_snapshot_of_the_whole_analysis(self):
        run = fake_refresh()
        self.assertIsNotNone(run)
        self.assertEqual(run.status, RefreshRun.Status.SUCCESS, run.error)
        self.assertEqual(run.coins, 6)
        self.assertIsNotNone(run.finished_at)
        self.assertEqual(run.charts_downloaded, 6)          # nothing cached yet: every history downloaded

        payload = run.snapshot.payload
        self.assertEqual(payload["meta"]["runId"], run.pk)  # pages compare this to detect new data
        self.assertEqual(len(payload["coins"]), 6)
        self.assertIn("lp", payload["coins"][0])            # live snapshot price
        self.assertEqual(payload["meta"]["totalOutliers"], run.outliers)

    def test_a_failing_pipeline_is_recorded_not_raised(self):
        with mock.patch.object(services, "run_pipeline", side_effect=RuntimeError("CoinGecko unreachable")):
            run = services.refresh_market_data(trigger=RefreshRun.Trigger.MANUAL)
        self.assertEqual(run.status, RefreshRun.Status.FAILED)
        self.assertIn("CoinGecko unreachable", run.error)
        self.assertFalse(DataSnapshot.objects.exists())

    def test_overlapping_refresh_is_skipped(self):
        RefreshRun.objects.create(status=RefreshRun.Status.RUNNING)
        with mock.patch.object(services, "run_pipeline") as pipeline:
            self.assertIsNone(services.refresh_market_data())
        pipeline.assert_not_called()

    def test_a_run_left_running_by_a_dead_process_stops_blocking(self):
        stale = RefreshRun.objects.create(status=RefreshRun.Status.RUNNING,
                                          started_at=timezone.now() - timedelta(hours=3))
        self.assertIsNone(services.active_run())             # too old to count as running
        self.assertEqual(services.mark_interrupted_runs(), 1)
        stale.refresh_from_db()
        self.assertEqual(stale.status, RefreshRun.Status.FAILED)

    @override_settings(CRYPTO_SNAPSHOTS_TO_KEEP=2)
    def test_old_snapshots_are_pruned_but_run_history_is_kept(self):
        for minutes in (45, 30, 15, 0):
            make_success(minutes)
        services.prune_snapshots()
        self.assertEqual(DataSnapshot.objects.count(), 2)
        self.assertEqual(RefreshRun.objects.count(), 4)
        newest = RefreshRun.objects.first()
        self.assertTrue(DataSnapshot.objects.filter(run=newest).exists())

    def test_refresh_uses_a_fresh_snapshot_and_cached_history(self):
        with mock.patch.object(services, "run_pipeline", side_effect=RuntimeError("stop")) as pipeline:
            services.refresh_market_data()
        cfg = pipeline.call_args.args[0]
        self.assertEqual(cfg.cache_minutes, 0)
        self.assertEqual(cfg.history_cache_minutes, settings.CRYPTO_HISTORY_CACHE_MINUTES)
        self.assertFalse(cfg.make_plots)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class SchedulerTests(TestCase):
    def test_refresh_job_runs_every_15_minutes_without_overlap(self):
        from apscheduler.schedulers.background import BackgroundScheduler

        sched = BackgroundScheduler(timezone="UTC")
        scheduler.add_jobs(sched, timezone.now())
        job = sched.get_job(scheduler.REFRESH_JOB_ID)
        self.assertEqual(job.trigger.interval, timedelta(minutes=15))
        self.assertEqual(job.max_instances, 1)
        self.assertTrue(job.coalesce)

    def test_stale_or_missing_data_refreshes_immediately(self):
        now = timezone.now()
        self.assertLess(scheduler.first_run_time(now) - now, timedelta(seconds=10))   # no data yet
        make_success(minutes_ago=40)
        self.assertLess(scheduler.first_run_time(now) - now, timedelta(seconds=10))   # older than 15 min

    def test_fresh_data_continues_the_existing_cadence(self):
        run = make_success(minutes_ago=5)
        self.assertEqual(scheduler.first_run_time(), run.finished_at + timedelta(minutes=15))

    def test_scheduler_only_starts_in_the_serving_runserver_process(self):
        from market.apps import MarketConfig

        with mock.patch("sys.argv", ["manage.py", "migrate"]):
            self.assertFalse(MarketConfig._is_serving_process())
        with mock.patch("sys.argv", ["manage.py", "runserver"]), mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("RUN_MAIN", None)
            self.assertFalse(MarketConfig._is_serving_process())   # autoreloader parent
        with mock.patch("sys.argv", ["manage.py", "runserver"]), mock.patch.dict("os.environ", {"RUN_MAIN": "true"}):
            self.assertTrue(MarketConfig._is_serving_process())


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class ApiTests(TestCase):
    def test_status_reports_last_success_and_next_run(self):
        run = make_success(minutes_ago=3)
        next_run = timezone.now() + timedelta(minutes=12)
        SchedulerState.objects.create(pk=1, heartbeat_at=timezone.now(), next_run_at=next_run, interval_minutes=15)
        body = self.client.get(reverse("market:status")).json()
        self.assertEqual(body["lastSuccess"]["id"], run.pk)
        self.assertFalse(body["running"])
        self.assertTrue(body["scheduler"]["alive"])
        self.assertIsNotNone(body["scheduler"]["nextRunAt"])
        self.assertEqual(body["intervalMinutes"], 15)

    def test_status_flags_a_dead_scheduler(self):
        SchedulerState.objects.create(pk=1, heartbeat_at=timezone.now() - timedelta(minutes=10),
                                      next_run_at=timezone.now(), interval_minutes=15)
        body = self.client.get(reverse("market:status")).json()
        self.assertFalse(body["scheduler"]["alive"])
        self.assertIsNone(body["scheduler"]["nextRunAt"])

    def test_data_script_serves_the_newest_snapshot(self):
        make_success(minutes_ago=20)
        newest = make_success(minutes_ago=1)
        response = self.client.get(reverse("market:data_js"))
        self.assertEqual(response["Content-Type"], "application/javascript; charset=utf-8")
        self.assertIn(f'"runId":{newest.pk}', response.content.decode())
        self.assertIn("no-cache", response.get("Cache-Control", ""))

    def test_data_json_is_unavailable_before_the_first_refresh(self):
        self.assertEqual(self.client.get(reverse("market:data_json")).status_code, 503)

    def test_refresh_endpoint_requires_post_and_csrf(self):
        self.assertEqual(self.client.get(reverse("market:refresh")).status_code, 405)
        strict = Client(enforce_csrf_checks=True)
        self.assertEqual(strict.post(reverse("market:refresh")).status_code, 403)

    def test_refresh_endpoint_starts_a_background_refresh(self):
        with mock.patch.object(services, "start_refresh_in_background") as start:
            response = self.client.post(reverse("market:refresh"))
        self.assertEqual(response.status_code, 202)
        start.assert_called_once()

    def test_refresh_endpoint_refuses_while_running_and_just_after_a_refresh(self):
        RefreshRun.objects.create(status=RefreshRun.Status.RUNNING)
        with mock.patch.object(services, "start_refresh_in_background") as start:
            self.assertEqual(self.client.post(reverse("market:refresh")).status_code, 409)
        start.assert_not_called()

        RefreshRun.objects.all().delete()
        make_success(minutes_ago=0)
        with mock.patch.object(services, "start_refresh_in_background") as start:
            self.assertEqual(self.client.post(reverse("market:refresh")).status_code, 429)
        start.assert_not_called()


class InterfaceTests(TestCase):
    def test_interfaces_load_live_data_and_the_refresh_indicator(self):
        for name in ("market:room", "market:deck"):
            html = self.client.get(reverse(name)).content.decode()
            self.assertIn(reverse("market:data_js"), html)
            self.assertIn("market/live.js", html)
            self.assertNotIn('src="analysis_data.js"', html)        # static data source replaced

    def test_index_lists_runs_and_interfaces(self):
        run = make_success(minutes_ago=2)
        response = self.client.get(reverse("market:index"))
        self.assertContains(response, "Market Refresh Server")
        self.assertContains(response, f"#{run.pk}")
        self.assertContains(response, reverse("market:room"))
        self.assertContains(response, "Every 15 min")
        self.assertIn("csrftoken", response.cookies)                  # needed by the refresh button

    def test_unknown_interface_is_404(self):
        from django.http import Http404
        from django.test import RequestFactory

        from market import views

        with self.assertRaises(Http404):
            views.interface(RequestFactory().get("/nope/"), "nope")
