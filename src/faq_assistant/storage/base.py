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
        """Create the collection if missing; fail if it exists with another embedding model.

        Returns:
            None. The collection is created or verified as a side effect.
        """
        ...

    async def list_collections(self) -> list[CollectionInfo]:
        """Return all collections.

        Returns:
            A list of CollectionInfo objects for all stored collections.
        """
        ...

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all of its items.

        Returns:
            None. The collection is deleted as a side effect.
        """
        ...

    async def get_content_hashes(self, collection: str) -> dict[str, str]:
        """Return ``{item_id: content_hash}`` for change detection.

        Returns:
            A dictionary mapping item IDs to their content hashes.
        """
        ...

    async def upsert(self, collection: str, items: Sequence[EmbeddedItem]) -> None:
        """Insert or update items.

        Returns:
            None. Items are upserted as a side effect.
        """
        ...

    async def delete_items(self, collection: str, item_ids: Sequence[str]) -> None:
        """Delete items by id.

        Returns:
            None. Items are deleted as a side effect.
        """
        ...

    async def search(
        self, collection: str, embedding: Sequence[float], top_k: int
    ) -> list[tuple[FAQItem, float]]:
        """Return up to ``top_k`` items ordered by descending cosine similarity.

        The similarity is the max over the question and document representations.

        Returns:
            A list of (FAQItem, similarity_score) tuples sorted by descending similarity.
        """
        ...

    async def ping(self) -> bool:
        """Return ``True`` if the store is reachable.

        Returns:
            True if the store is reachable, False otherwise.
        """
        ...

    async def close(self) -> None:
        """Release resources.

        Returns:
            None. Resources are released as a side effect.
        """
        ...
