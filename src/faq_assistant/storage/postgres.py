"""PostgreSQL + pgvector implementation of :class:`~faq_assistant.storage.base.VectorStore`."""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from faq_assistant.domain.errors import (
    CollectionNotFoundError,
    EmbeddingModelMismatchError,
    VectorStoreError,
)
from faq_assistant.domain.models import FAQItem
from faq_assistant.storage.base import CollectionInfo, EmbeddedItem

logger = logging.getLogger(__name__)


def schema_sql(dimensions: int) -> str:
    """DDL for the knowledge-base schema (idempotent).

    Args:
        dimensions: Dimensionality of the embedding vectors.

    Returns:
        A SQL DDL string for creating the schema.
    """
    dims = int(dimensions)  # interpolated into DDL: force an int to rule out injection
    return f"""
    CREATE TABLE IF NOT EXISTS collections (
        name                 TEXT PRIMARY KEY,
        embedding_model      TEXT        NOT NULL,
        embedding_dimensions INTEGER     NOT NULL,
        created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS faq_items (
        id                 TEXT PRIMARY KEY,
        collection         TEXT        NOT NULL REFERENCES collections(name) ON DELETE CASCADE,
        question           TEXT        NOT NULL,
        answer             TEXT        NOT NULL,
        category           TEXT        NOT NULL,
        metadata           JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
        content_hash       TEXT        NOT NULL,
        question_embedding VECTOR({dims}) NOT NULL,
        document_embedding VECTOR({dims}) NOT NULL,
        created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS faq_items_collection_idx ON faq_items (collection);
    CREATE INDEX IF NOT EXISTS faq_items_category_idx ON faq_items (collection, category);
    CREATE INDEX IF NOT EXISTS faq_items_question_hnsw
        ON faq_items USING hnsw (question_embedding vector_cosine_ops);
    CREATE INDEX IF NOT EXISTS faq_items_document_hnsw
        ON faq_items USING hnsw (document_embedding vector_cosine_ops);
    """


# Two index-backed kNN scans (one per representation), merged with max-sim.
_SEARCH_SQL = """
WITH candidates AS (
    (SELECT id, 1 - (question_embedding <=> %(embedding)s) AS similarity
       FROM faq_items WHERE collection = %(collection)s
      ORDER BY question_embedding <=> %(embedding)s LIMIT %(top_k)s)
    UNION ALL
    (SELECT id, 1 - (document_embedding <=> %(embedding)s) AS similarity
       FROM faq_items WHERE collection = %(collection)s
      ORDER BY document_embedding <=> %(embedding)s LIMIT %(top_k)s)
)
SELECT f.id, f.question, f.answer, f.category, f.metadata, MAX(c.similarity) AS similarity
  FROM candidates c JOIN faq_items f ON f.id = c.id
 GROUP BY f.id
 ORDER BY similarity DESC
 LIMIT %(top_k)s
"""

_UPSERT_SQL = """
INSERT INTO faq_items (id, collection, question, answer, category, metadata, content_hash,
                       question_embedding, document_embedding)
VALUES (%(id)s, %(collection)s, %(question)s, %(answer)s, %(category)s, %(metadata)s,
        %(content_hash)s, %(question_embedding)s, %(document_embedding)s)
ON CONFLICT (id) DO UPDATE SET
    question = EXCLUDED.question,
    answer = EXCLUDED.answer,
    category = EXCLUDED.category,
    metadata = EXCLUDED.metadata,
    content_hash = EXCLUDED.content_hash,
    question_embedding = EXCLUDED.question_embedding,
    document_embedding = EXCLUDED.document_embedding,
    updated_at = now()
"""


class PostgresVectorStore:
    """pgvector-backed store with HNSW cosine indexes."""

    def __init__(self, dsn: str, dimensions: int, *, min_size: int = 1, max_size: int = 10):
        """Initialize the PostgreSQL vector store.

        Args:
            dsn: PostgreSQL connection string.
            dimensions: Embedding vector dimensionality.
            min_size: Minimum pool connection count.
            max_size: Maximum pool connection count.
        """
        self._dsn = dsn
        self._dimensions = dimensions
        self._pool = AsyncConnectionPool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            open=False,
            configure=register_vector_async,
            kwargs={"row_factory": dict_row},
        )

    async def initialize(self) -> None:
        """Create the extension and schema, then open the connection pool.

        Returns:
            None. The connection pool is opened as a side effect.
        """
        try:
            async with await psycopg.AsyncConnection.connect(self._dsn, autocommit=True) as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.execute(schema_sql(self._dimensions).encode())
            await self._pool.open(wait=True)
        except psycopg.Error as exc:
            msg = f"Could not initialise PostgreSQL vector store: {exc}"
            raise VectorStoreError(msg) from exc

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[psycopg.AsyncConnection[dict[str, Any]]]:
        """Context manager for acquiring a connection from the pool.

        Returns:
            An async context manager yielding a PostgreSQL connection.
        """
        try:
            async with self._pool.connection() as conn:
                yield conn
        except psycopg.Error as exc:
            logger.exception("PostgreSQL operation failed")
            msg = f"Vector store operation failed: {exc.__class__.__name__}"
            raise VectorStoreError(msg) from exc

    async def ensure_collection(self, name: str, embedding_model: str, dimensions: int) -> None:
        """Create the collection if missing; fail if it exists with another embedding model.

        Returns:
            None. The collection is created or verified as a side effect.
        """
        if dimensions != self._dimensions:
            msg = f"Store is configured for {self._dimensions}-d vectors, got {dimensions}"
            raise EmbeddingModelMismatchError(msg)
        async with self._connection() as conn:
            await conn.execute(
                "INSERT INTO collections (name, embedding_model, embedding_dimensions) "
                "VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING",
                (name, embedding_model, dimensions),
            )
            cursor = await conn.execute(
                "SELECT embedding_model FROM collections WHERE name = %s", (name,)
            )
            row = await cursor.fetchone()
        if row is not None and row["embedding_model"] != embedding_model:
            msg = (
                f"Collection {name!r} was embedded with {row['embedding_model']!r}, "
                f"not {embedding_model!r}. Re-create it to switch models."
            )
            raise EmbeddingModelMismatchError(msg)

    async def list_collections(self) -> list[CollectionInfo]:
        """Return all collections with their item counts.

        Returns:
            A list of CollectionInfo objects for all stored collections.
        """
        async with self._connection() as conn:
            cursor = await conn.execute(
                "SELECT c.name, c.embedding_model, c.embedding_dimensions, COUNT(f.id) AS n "
                "FROM collections c LEFT JOIN faq_items f ON f.collection = c.name "
                "GROUP BY c.name ORDER BY c.name"
            )
            rows = await cursor.fetchall()
        return [
            CollectionInfo(r["name"], r["embedding_model"], r["embedding_dimensions"], r["n"])
            for r in rows
        ]

    async def delete_collection(self, name: str) -> None:
        """Delete a collection (items cascade).

        Returns:
            None. The collection is deleted as a side effect.
        """
        async with self._connection() as conn:
            await conn.execute("DELETE FROM collections WHERE name = %s", (name,))

    async def get_content_hashes(self, collection: str) -> dict[str, str]:
        """Return ``{item_id: content_hash}`` for change detection.

        Returns:
            A dictionary mapping item IDs to their content hashes.
        """
        async with self._connection() as conn:
            cursor = await conn.execute(
                "SELECT id, content_hash FROM faq_items WHERE collection = %s", (collection,)
            )
            rows = await cursor.fetchall()
        return {r["id"]: r["content_hash"] for r in rows}

    async def upsert(self, collection: str, items: Sequence[EmbeddedItem]) -> None:
        """Insert or update items in a single transaction.

        Returns:
            None. Items are upserted as a side effect.
        """
        if not items:
            return
        params = [
            {
                "id": e.item.id,
                "collection": collection,
                "question": e.item.question,
                "answer": e.item.answer,
                "category": e.item.category,
                "metadata": Jsonb(e.item.metadata),
                "content_hash": e.content_hash,
                "question_embedding": np.asarray(e.question_embedding, dtype=np.float32),
                "document_embedding": np.asarray(e.document_embedding, dtype=np.float32),
            }
            for e in items
        ]
        async with self._connection() as conn, conn.transaction(), conn.cursor() as cursor:
            await cursor.executemany(_UPSERT_SQL, params)

    async def delete_items(self, collection: str, item_ids: Sequence[str]) -> None:
        """Delete items by id.

        Returns:
            None. Items are deleted as a side effect.
        """
        if not item_ids:
            return
        async with self._connection() as conn:
            await conn.execute(
                "DELETE FROM faq_items WHERE collection = %s AND id = ANY(%s)",
                (collection, list(item_ids)),
            )

    async def search(
        self, collection: str, embedding: Sequence[float], top_k: int
    ) -> list[tuple[FAQItem, float]]:
        """Return up to ``top_k`` items by max cosine similarity over both representations.

        Returns:
            A list of (FAQItem, similarity_score) tuples sorted by descending similarity.
        """
        async with self._connection() as conn:
            cursor = await conn.execute("SELECT 1 FROM collections WHERE name = %s", (collection,))
            if await cursor.fetchone() is None:
                msg = f"Collection {collection!r} does not exist"
                raise CollectionNotFoundError(msg)
            cursor = await conn.execute(
                _SEARCH_SQL,
                {
                    "embedding": np.asarray(embedding, dtype=np.float32),
                    "collection": collection,
                    "top_k": top_k,
                },
            )
            rows = await cursor.fetchall()
        return [
            (
                FAQItem(
                    id=r["id"],
                    question=r["question"],
                    answer=r["answer"],
                    category=r["category"],
                    metadata=r["metadata"],
                ),
                float(r["similarity"]),
            )
            for r in rows
        ]

    async def ping(self) -> bool:
        """Return ``True`` if the database answers a trivial query.

        Returns:
            True if the database is reachable, False on error.
        """
        try:
            async with self._connection() as conn:
                await conn.execute("SELECT 1")
        except VectorStoreError:
            return False
        return True

    async def close(self) -> None:
        """Close the connection pool.

        Returns:
            None. The connection pool is closed as a side effect.
        """
        await self._pool.close()
