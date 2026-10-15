from django.db import models
from django.utils import timezone


class RefreshRun(models.Model):
    """One execution of the refresh job, successful or not."""

    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"

    class Trigger(models.TextChoices):
        SCHEDULE = "schedule", "Schedule"
        MANUAL = "manual", "Manual"
        COMMAND = "command", "Command"

    started_at = models.DateTimeField(default=timezone.now, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.RUNNING, db_index=True)
    trigger = models.CharField(max_length=10, choices=Trigger.choices, default=Trigger.SCHEDULE)
    coins = models.PositiveIntegerField(null=True, blank=True)
    observations = models.PositiveIntegerField(null=True, blank=True)
    outliers = models.PositiveIntegerField(null=True, blank=True)
    charts_downloaded = models.PositiveIntegerField(
        null=True, blank=True, help_text="Coin histories fetched from the API (the rest came from cache).")
    window_end = models.DateTimeField(null=True, blank=True)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"Run #{self.pk} {self.status} ({self.started_at:%Y-%m-%d %H:%M} UTC)"

    @property
    def duration_seconds(self) -> float | None:
        if not self.finished_at:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class DataSnapshot(models.Model):
    """The UI payload produced by a successful run."""

    run = models.OneToOneField(RefreshRun, on_delete=models.CASCADE, related_name="snapshot")
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    payload = models.JSONField()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Snapshot for run #{self.run_id}"


class SchedulerState(models.Model):
    """Heartbeat written by whichever process runs the scheduler.

    Lets the web process report the next refresh time even when the scheduler
    runs separately (`manage.py runscheduler`). A single row, pk=1.
    """

    heartbeat_at = models.DateTimeField()
    next_run_at = models.DateTimeField(null=True, blank=True)
    interval_minutes = models.PositiveIntegerField()
    process = models.CharField(max_length=80, blank=True)

    def __str__(self) -> str:
        return f"Scheduler heartbeat {self.heartbeat_at:%H:%M:%S} UTC"
