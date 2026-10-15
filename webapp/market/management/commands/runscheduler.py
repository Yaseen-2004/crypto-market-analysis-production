from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = ("Run the market data refresh loop as its own process. Use with "
            "CRYPTO_SCHEDULER_AUTOSTART=0 on the web server so only one scheduler runs.")

    def handle(self, *args, **options):
        from market import scheduler

        self.stdout.write("Refresh scheduler running. Press Ctrl+C to stop.")
        try:
            scheduler.start(blocking=True)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
            self.stdout.write("Scheduler stopped.")
