"""Celery application for asynchronous embedding jobs."""

from celery import Celery

from faq_assistant.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "faq_assistant",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
    include=["faq_assistant.worker.tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,  # re-deliver if a worker dies mid-job (sync is idempotent)
    worker_prefetch_multiplier=1,
    result_expires=24 * 3600,
    task_track_started=True,
)
