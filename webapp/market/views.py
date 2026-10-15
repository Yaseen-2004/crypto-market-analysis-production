from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from django.templatetags.static import static
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from src.ui_payload import to_script

from . import services
from .models import RefreshRun

UI_DIR = settings.PROJECT_ROOT / "ui"
INTERFACES = {
    "room": ("risk_room.html", "Risk Room", "Big-screen control room"),
    "deck": ("volatility_deck.html", "Volatility Deck", "Phone-framed app"),
}
STATIC_DATA_TAG = '<script src="analysis_data.js"></script>'
MANUAL_REFRESH_COOLDOWN = timedelta(seconds=60)


def _live_tags() -> str:
    """Script tags that point a static UI page at the live data and the refresh indicator."""
    return (
        f'<script src="{reverse("market:data_js")}"></script>\n'
        f'<script src="{static("market/live.js")}" defer'
        f' data-status-url="{reverse("market:status")}"'
        f' data-refresh-url="{reverse("market:refresh")}"></script>'
    )


@require_GET
@ensure_csrf_cookie
@never_cache
def index(request):
    runs = RefreshRun.objects.all()[:12]
    return render(request, "market/index.html", {
        "status": services.status(),
        "runs": runs,
        "interfaces": [(key, title, blurb) for key, (_, title, blurb) in INTERFACES.items()],
        "interval": settings.CRYPTO_REFRESH_INTERVAL_MINUTES,
        "history_cache": int(settings.CRYPTO_HISTORY_CACHE_MINUTES),
        "live_tags": _live_tags(),
    })


@require_GET
@ensure_csrf_cookie
@never_cache
def interface(request, name: str):
    """Serve a UI page from ui/ with its data source swapped for the live endpoint.

    The files stay plain HTML (they still open from disk), so they are read and
    patched rather than rendered as templates - their CSS and JS braces would
    otherwise collide with template syntax.
    """
    if name not in INTERFACES:
        raise Http404("Unknown interface")
    html = (UI_DIR / INTERFACES[name][0]).read_text(encoding="utf-8")
    if STATIC_DATA_TAG not in html:
        raise Http404("Interface page is missing its data script tag")
    return HttpResponse(html.replace(STATIC_DATA_TAG, _live_tags()), content_type="text/html; charset=utf-8")


@require_GET
@never_cache
def data_js(request):
    """Latest payload as `window.CRYPTO_DATA=...;` (falls back to the static build before the first refresh)."""
    snapshot = services.latest_snapshot()
    if snapshot is not None:
        body = to_script(snapshot.payload)
    else:
        fallback = UI_DIR / "analysis_data.js"
        body = fallback.read_text(encoding="utf-8") if fallback.exists() else "window.CRYPTO_DATA=null;"
    return HttpResponse(body, content_type="application/javascript; charset=utf-8")


@require_GET
@never_cache
def data_json(request):
    snapshot = services.latest_snapshot()
    if snapshot is None:
        return JsonResponse({"error": "No refresh has completed yet."}, status=503)
    return JsonResponse(snapshot.payload)


@require_GET
@never_cache
def status_json(request):
    return JsonResponse(services.status())


@require_POST
def refresh(request):
    """Start a refresh now. CSRF-protected: it triggers external API calls."""
    if services.active_run():
        return JsonResponse({"started": False, "reason": "A refresh is already running."}, status=409)
    last = RefreshRun.objects.first()
    if last and last.finished_at and timezone.now() - last.finished_at < MANUAL_REFRESH_COOLDOWN:
        wait = int((MANUAL_REFRESH_COOLDOWN - (timezone.now() - last.finished_at)).total_seconds()) + 1
        return JsonResponse({"started": False, "reason": f"Refreshed moments ago - try again in {wait}s."},
                            status=429)
    services.start_refresh_in_background(trigger=RefreshRun.Trigger.MANUAL)
    return JsonResponse({"started": True}, status=202)
