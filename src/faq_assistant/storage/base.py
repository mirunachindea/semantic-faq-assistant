"""Vector store abstraction.

The application depends on this protocol only, so the backing store (in-memory, pgvector,
a managed vector DB, ...) can be swapped without touching retrieval or routing code.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from faq_assistant.domain.models import FAQItem


@dataclass(frozen=True, slots=True)
class EmbeddedItem:
    """An FAQ item with its two vector representations and change-detection hash."""

    item: FAQItem
    content_hash: str
    question_embedding: Sequence[float]
    document_embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """Summary of a stored collection."""

    name: str
    embedding_model: str
    embedding_dimensions: int
    item_count: int


class VectorStore(Protocol):
    """Persistence + nearest-neighbour search for embedded FAQ items."""

    async def ensure_collection(self, name: str, embedding_model: str, dimensions: int) -> None:
        """Create the collection if missing; fail if it exists with another embedding model."""
        ...

    async def list_collections(self) -> list[CollectionInfo]:
        """Return all collections."""
        ...

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all of its items."""
        ...

    async def get_content_hashes(self, collection: str) -> dict[str, str]:
        """Return ``{item_id: content_hash}`` for change detection."""
        ...

    async def upsert(self, collection: str, items: Sequence[EmbeddedItem]) -> None:
        """Insert or update items."""
        ...

    async def delete_items(self, collection: str, item_ids: Sequence[str]) -> None:
        """Delete items by id."""
        ...

    async def search(
        self, collection: str, embedding: Sequence[float], top_k: int
    ) -> list[tuple[FAQItem, float]]:
        """Return up to ``top_k`` items ordered by descending cosine similarity.

        The similarity is the max over the question and document representations.
        """
        ...

    async def ping(self) -> bool:
        """Return ``True`` if the store is reachable."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
