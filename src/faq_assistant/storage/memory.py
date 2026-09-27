"""In-process vector store (numpy). Used for tests and dependency-free local runs."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from faq_assistant.domain.errors import CollectionNotFoundError, EmbeddingModelMismatchError
from faq_assistant.domain.models import FAQItem
from faq_assistant.storage.base import CollectionInfo, EmbeddedItem

FloatArray = npt.NDArray[np.float64]


def _unit(vector: Sequence[float]) -> FloatArray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


@dataclass(slots=True)
class _Collection:
    embedding_model: str
    dimensions: int
    items: dict[str, EmbeddedItem] = field(default_factory=dict)


class InMemoryVectorStore:
    """Brute-force cosine search; adequate for knowledge bases of a few thousand items."""

    def __init__(self) -> None:
        """Initialize the in-memory vector store."""
        self._collections: dict[str, _Collection] = {}

    def _get(self, name: str) -> _Collection:
        try:
            return self._collections[name]
        except KeyError as exc:
            msg = f"Collection {name!r} does not exist"
            raise CollectionNotFoundError(msg) from exc

    async def ensure_collection(self, name: str, embedding_model: str, dimensions: int) -> None:
        """Create the collection if missing; fail if it exists with another embedding model.

        Returns:
            None. The collection is created or verified as a side effect.
        """
        existing = self._collections.get(name)
        if existing is None:
            self._collections[name] = _Collection(embedding_model, dimensions)
        elif existing.embedding_model != embedding_model:
            msg = (
                f"Collection {name!r} was embedded with {existing.embedding_model!r}, "
                f"not {embedding_model!r}"
            )
            raise EmbeddingModelMismatchError(msg)

    async def list_collections(self) -> list[CollectionInfo]:
        """Return all collections.

        Returns:
            A list of CollectionInfo objects for all stored collections.
        """
        return [
            CollectionInfo(name, c.embedding_model, c.dimensions, len(c.items))
            for name, c in sorted(self._collections.items())
        ]

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all of its items.

        Returns:
            None. The collection is deleted as a side effect.
        """
        self._collections.pop(name, None)

    async def get_content_hashes(self, collection: str) -> dict[str, str]:
        """Return ``{item_id: content_hash}``.

        Returns:
            A dictionary mapping item IDs to their content hashes.
        """
        items = self._collections[collection].items if collection in self._collections else {}
        return {item_id: embedded.content_hash for item_id, embedded in items.items()}

    async def upsert(self, collection: str, items: Sequence[EmbeddedItem]) -> None:
        """Insert or update items.

        Returns:
            None. Items are upserted as a side effect.
        """
        target = self._get(collection)
        for embedded in items:
            target.items[embedded.item.id] = embedded

    async def delete_items(self, collection: str, item_ids: Sequence[str]) -> None:
        """Delete items by id.

        Returns:
            None. Items are deleted as a side effect.
        """
        target = self._get(collection)
        for item_id in item_ids:
            target.items.pop(item_id, None)

    async def search(
        self, collection: str, embedding: Sequence[float], top_k: int
    ) -> list[tuple[FAQItem, float]]:
        """Return the ``top_k`` most similar items (max-sim over both representations).

        Returns:
            A list of (FAQItem, similarity_score) tuples sorted by descending similarity.
        """
        target = self._get(collection)
        if not target.items:
            return []
        query = _unit(embedding)
        entries = list(target.items.values())
        questions = np.stack([_unit(e.question_embedding) for e in entries])
        documents = np.stack([_unit(e.document_embedding) for e in entries])
        scores = np.maximum(questions @ query, documents @ query)
        order = np.argsort(-scores)[:top_k]
        return [(entries[i].item, float(scores[i])) for i in order]

    async def ping(self) -> bool:
        """Always reachable.

        Returns:
            Always True for the in-memory store.
        """
        return True

    async def close(self) -> None:
        """Nothing to release.

        Returns:
            None. No resources to close for the in-memory store.
        """
