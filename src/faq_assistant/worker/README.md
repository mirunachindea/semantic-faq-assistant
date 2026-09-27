# worker

Celery worker for asynchronous knowledge-base ingestion.

- `celery_app.py` — the Celery application.
- `tasks.py` — `sync_collection_task`, which curates and incrementally embeds items into a collection (queued by the admin ingestion endpoint).
