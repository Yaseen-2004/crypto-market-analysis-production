from django.core.management.base import BaseCommand, CommandError

from market import services
from market.models import RefreshRun


class Command(BaseCommand):
    help = "Refresh the market data once, now (also usable from cron instead of the scheduler)."

    def add_arguments(self, parser):
        parser.add_argument("--history-cache-minutes", type=float, default=None,
                            help="Reuse downloaded histories younger than this (default: settings value).")

    def handle(self, *args, **options):
        overrides = {}
        if options["history_cache_minutes"] is not None:
            overrides["history_cache_minutes"] = options["history_cache_minutes"]
        run = services.refresh_market_data(trigger=RefreshRun.Trigger.COMMAND, **overrides)
        if run is None:
            raise CommandError("Another refresh is already running.")
        if run.status != RefreshRun.Status.SUCCESS:
            raise CommandError(f"Refresh #{run.pk} failed: {run.error}")
        self.stdout.write(self.style.SUCCESS(
            f"Refresh #{run.pk} done in {run.duration_seconds:.0f}s: {run.coins} coins, "
            f"{run.outliers} outliers, {run.charts_downloaded} histories downloaded."))
