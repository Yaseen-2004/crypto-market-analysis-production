from django.contrib import admin

from .models import DataSnapshot, RefreshRun, SchedulerState


@admin.register(RefreshRun)
class RefreshRunAdmin(admin.ModelAdmin):
    list_display = ("id", "started_at", "status", "trigger", "duration", "coins", "outliers",
                    "charts_downloaded", "window_end")
    list_filter = ("status", "trigger")
    readonly_fields = [f.name for f in RefreshRun._meta.fields]

    @admin.display(description="Duration")
    def duration(self, obj):
        seconds = obj.duration_seconds
        return "—" if seconds is None else f"{seconds:.0f}s"


@admin.register(DataSnapshot)
class DataSnapshotAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "created_at")
    exclude = ("payload",)  # ~650 KB of JSON is not something to render in a form


@admin.register(SchedulerState)
class SchedulerStateAdmin(admin.ModelAdmin):
    list_display = ("heartbeat_at", "next_run_at", "interval_minutes", "process")
