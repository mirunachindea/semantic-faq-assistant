"""Celery tasks."""

import asyncio
import logging
from typing import Any

from faq_assistant.config import get_settings
from faq_assistant.container import create_indexer, create_store
from faq_assistant.domain.errors import UpstreamRateLimitError
from faq_assistant.knowledge_base.loader import curate_entries, parse_knowledge_base
from faq_assistant.llm.factory import build_embeddings
from faq_assistant.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _sync(collection: str, raw_items: list[dict[str, Any]], *, prune: bool) -> dict[str, Any]:
    settings = get_settings()
    source = parse_knowledge_base({"knowledge_base_items": raw_items})
    curated = curate_entries(source.knowledge_base_items, collection)
    store = await create_store(settings)
    try:
        indexer = create_indexer(settings, store, build_embeddings(settings))
        report = await indexer.sync(collection, curated.items, prune=prune)
    finally:
        await store.close()
    return {
        **report.as_dict(),
        "rejected": [
            {"question": r.question, "issues": [i.code for i in r.issues]} for r in curated.rejected
        ],
    }


@celery_app.task(  # type: ignore[untyped-decorator]
    name="faq.sync_collection",
    bind=True,
    autoretry_for=(UpstreamRateLimitError,),
    retry_backoff=True,
    retry_backoff_max=120,
    max_retries=5,
)
def sync_collection_task(
    self: Any,
    collection: str,
    raw_items: list[dict[str, Any]],
    *,
    prune: bool = False,
) -> dict[str, Any]:
    """Curate and incrementally embed ``raw_items`` into ``collection``.

    Idempotent: re-running with the same payload embeds nothing (content hashing), which makes
    ``acks_late`` redelivery and rate-limit retries safe.
    """
    logger.info("Sync task started for %r (%d items)", collection, len(raw_items))
    return asyncio.run(_sync(collection, raw_items, prune=prune))
