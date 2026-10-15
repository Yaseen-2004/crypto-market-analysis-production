from django.urls import path

from . import views

app_name = "market"

urlpatterns = [
    path("", views.index, name="index"),
    path("room/", views.interface, {"name": "room"}, name="room"),
    path("deck/", views.interface, {"name": "deck"}, name="deck"),
    path("api/data.js", views.data_js, name="data_js"),
    path("api/data.json", views.data_json, name="data_json"),
    path("api/status/", views.status_json, name="status"),
    path("api/refresh/", views.refresh, name="refresh"),
]
