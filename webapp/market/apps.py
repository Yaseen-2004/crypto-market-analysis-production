import os
import sys

from django.apps import AppConfig
from django.conf import settings


class MarketConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "market"
    verbose_name = "Crypto market data"

    def ready(self) -> None:
        if not settings.CRYPTO_SCHEDULER_AUTOSTART or not self._is_serving_process():
            return
        from . import scheduler

        scheduler.start()

    @staticmethod
    def _is_serving_process() -> bool:
        """True only in the process that actually serves `runserver` requests.

        The autoreloader runs a watcher parent plus a child (RUN_MAIN=true);
        starting in both would create two schedulers. Other commands - migrate,
        test, shell - never start one.
        """
        argv = sys.argv
        if len(argv) < 2 or argv[1] != "runserver":
            return False
        return "--noreload" in argv or os.environ.get("RUN_MAIN") == "true"
