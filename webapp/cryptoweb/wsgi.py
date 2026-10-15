"""WSGI entry point.

A production WSGI server runs several worker processes, so do not let each one
start a scheduler: set CRYPTO_SCHEDULER_AUTOSTART=0 there and run
`python manage.py runscheduler` as a single separate process.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "cryptoweb.settings")
application = get_wsgi_application()
