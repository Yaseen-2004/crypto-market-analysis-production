"""Django settings for the crypto market web app.

Local development works with no configuration. For anything reachable by other
people, set DJANGO_SECRET_KEY, DJANGO_DEBUG=0 and DJANGO_ALLOWED_HOSTS.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # webapp/
PROJECT_ROOT = BASE_DIR.parent                              # crypto_market_analysis/

# The analysis pipeline lives in PROJECT_ROOT/src; make it importable as `src`.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "django-insecure-local-development-only-change-me")
DEBUG = env_flag("DJANGO_DEBUG", True)
ALLOWED_HOSTS = [
    host.strip() for host in os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "market",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "cryptoweb.urls"
WSGI_APPLICATION = "cryptoweb.wsgi.application"
ASGI_APPLICATION = "cryptoweb.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        # The refresh job writes from a background thread while requests read;
        # wait for the lock instead of failing immediately.
        "OPTIONS": {"timeout": 20},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Market data refresh
# ---------------------------------------------------------------------------
# How often the scheduler refreshes the analysis.
CRYPTO_REFRESH_INTERVAL_MINUTES = int(os.environ.get("CRYPTO_REFRESH_MINUTES", "15"))
# CoinGecko's multi-day history is hourly, so it only gains a bar once an hour.
# Reusing it for up to this long keeps each 15-minute cycle to a single API call
# (the live market snapshot); the 100-call history download happens about hourly.
CRYPTO_HISTORY_CACHE_MINUTES = float(os.environ.get("CRYPTO_HISTORY_CACHE_MINUTES", "50"))
CRYPTO_TOP_N = int(os.environ.get("CRYPTO_TOP_N", "100"))
CRYPTO_DAYS = int(os.environ.get("CRYPTO_DAYS", "5"))
# A run still marked "running" after this long is treated as dead (process killed).
CRYPTO_RUN_TIMEOUT_MINUTES = int(os.environ.get("CRYPTO_RUN_TIMEOUT_MINUTES", "90"))
# Payloads are ~650 KB each; keep only the most recent few.
CRYPTO_SNAPSHOTS_TO_KEEP = int(os.environ.get("CRYPTO_SNAPSHOTS_TO_KEEP", "8"))
# Start the scheduler inside `manage.py runserver`. Turn off when running
# `manage.py runscheduler` as its own process, so only one scheduler exists.
CRYPTO_SCHEDULER_AUTOSTART = env_flag("CRYPTO_SCHEDULER_AUTOSTART", True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "datefmt": "%H:%M:%S"},
    },
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain"}},
    "loggers": {
        "market": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "crypto_analysis": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "src": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "apscheduler": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    },
}
