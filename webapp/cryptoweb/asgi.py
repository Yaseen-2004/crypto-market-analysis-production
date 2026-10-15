"""ASGI entry point. See wsgi.py for how to run the scheduler in production."""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cryptoweb.settings")
application = get_asgi_application()
