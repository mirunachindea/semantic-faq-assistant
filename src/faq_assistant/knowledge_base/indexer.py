"""Incremental, token-efficient embedding sync.

Each item's ``content_hash`` covers the embedding model and both texts that get embedded.
A sync only calls the embedding API for items whose hash is new or changed; unchanged items
cost zero tokens. Removing stale items is opt-in (``prune``) so partial updates never delete
existing data by accident.
"""

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings

from faq_assistant.domain.errors import UpstreamServiceError
from faq_assistant.domain.models import FAQItem
from faq_assistant.llm.errors import upstream_call
from faq_assistant.storage.base import EmbeddedItem, VectorStore

logger = logging.getLogger(__name__)


def content_hash(item: FAQItem, embedding_model: str) -> str:
    """Hash of everything that influences an item's stored representation.

    Args:
        item: The FAQ item to hash.
        embedding_model: The embedding model name to include in the hash.

    Returns:
        A SHA256 hex digest representing the item's content.
    """
    payload = "\x1f".join(
        (embedding_model, item.question_text, item.document_text, item.question, item.category)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class SyncReport:
    """Summary of a sync run."""

    collection: str
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: int = 0
    deleted: list[str] = field(default_factory=list)
    embedded_texts: int = 0

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable representation of the sync report.

        Returns:
            A dictionary with sync statistics (added, updated, unchanged, deleted counts).
        """
        return {
            "collection": self.collection,
            "added": len(self.added),
            "updated": len(self.updated),
            "unchanged": self.unchanged,
            "deleted": len(self.deleted),
            "embedded_texts": self.embedded_texts,
        }


class EmbeddingIndexer:
    """Keeps a vector-store collection in sync with a list of curated FAQ items."""

    def __init__(
        self,
        store: VectorStore,
        embeddings: Embeddings,
        *,
        embedding_model: str,
        dimensions: int,
        batch_size: int = 128,
    ):
        """Initialize the EmbeddingIndexer.

        Args:
            store: Vector store to sync items into.
            embeddings: Embeddings provider for generating vectors.
            embedding_model: Name of the embedding model for hash tracking.
            dimensions: Expected dimensionality of embedding vectors.
            batch_size: Number of items to process in each embedding batch.
        """
        self._store = store
        self._embeddings = embeddings
        self._embedding_model = embedding_model
        self._dimensions = dimensions
        self._batch_size = batch_size

    async def sync(
        self,
        collection: str,
        items: Sequence[FAQItem],
        *,
        prune: bool = False,
        force: bool = False,
    ) -> SyncReport:
        """Embed new/changed items and upsert them.

        Args:
            collection: Target collection (created if missing).
            items: Curated items that should be present in the collection.
            prune: Delete stored items that are not in ``items``.
            force: Re-embed everything, ignoring stored hashes.

        Returns:
            A SyncReport summarizing the operations performed.
        """
        await self._store.ensure_collection(collection, self._embedding_model, self._dimensions)
        existing = await self._store.get_content_hashes(collection)
        report = SyncReport(collection=collection)

        pending: list[tuple[FAQItem, str]] = []
        for item in items:
            digest = content_hash(item, self._embedding_model)
            stored = existing.get(item.id)
            if stored == digest and not force:
                report.unchanged += 1
                continue
            (report.added if stored is None else report.updated).append(item.id)
            pending.append((item, digest))

        for start in range(0, len(pending), self._batch_size):
            batch = pending[start : start + self._batch_size]
            await self._store.upsert(collection, await self._embed_batch(batch))
            report.embedded_texts += 2 * len(batch)

        if prune:
            wanted = {item.id for item in items}
            report.deleted = [item_id for item_id in existing if item_id not in wanted]
            await self._store.delete_items(collection, report.deleted)

        logger.info("Sync finished: %s", report.as_dict())
        return report

    async def _embed_batch(self, batch: Sequence[tuple[FAQItem, str]]) -> list[EmbeddedItem]:
        """Generate embeddings for a batch of items.

        Args:
            batch: Sequence of tuples containing FAQItem and its content hash.

        Returns:
            A list of EmbeddedItem objects with generated embeddings.
        """
        texts = [item.question_text for item, _ in batch] + [
            item.document_text for item, _ in batch
        ]
        async with upstream_call("document embedding"):
            vectors = await self._embeddings.aembed_documents(texts)
        if len(vectors) != len(texts) or any(len(v) != self._dimensions for v in vectors):
            msg = "Embedding provider returned vectors of unexpected count or dimensionality"
            raise UpstreamServiceError(msg)
        n = len(batch)
        return [
            EmbeddedItem(
                item=item,
                content_hash=digest,
                question_embedding=vectors[i],
                document_embedding=vectors[n + i],
            )
            for i, (item, digest) in enumerate(batch)
        ]
