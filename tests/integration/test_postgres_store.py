"""PostgreSQL/pgvector store tests.

Skipped unless ``TEST_DATABASE_URL`` points at a disposable database with pgvector, e.g.::

    docker compose up -d db
    TEST_DATABASE_URL=postgresql://faq:faq@localhost:5432/faq uv run pytest -m postgres
"""

import os
import uuid
from collections.abc import AsyncIterator

import pytest

from faq_assistant.domain.errors import CollectionNotFoundError, EmbeddingModelMismatchError
from faq_assistant.domain.models import FAQItem
from faq_assistant.knowledge_base.indexer import EmbeddingIndexer
from faq_assistant.storage.postgres import PostgresVectorStore
from tests.fakes import DIMENSIONS, BagOfWordsEmbeddings

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set"),
]

ITEMS = [
    FAQItem(id=f"t-{i}", question=q, answer="An answer long enough.", category="test")
    for i, q in enumerate(["Cancel subscription", "Download invoices", "Enable 2FA"])
]


@pytest.fixture
async def store() -> AsyncIterator[PostgresVectorStore]:
    assert DSN is not None
    pg = PostgresVectorStore(DSN, DIMENSIONS)
    await pg.initialize()
    yield pg
    await pg.close()


@pytest.fixture
def collection() -> str:
    return f"test-{uuid.uuid4().hex[:8]}"


async def test_sync_search_and_prune_roundtrip(store: PostgresVectorStore, collection: str) -> None:
    embeddings = BagOfWordsEmbeddings()
    indexer = EmbeddingIndexer(store, embeddings, embedding_model="fake", dimensions=DIMENSIONS)
    try:
        report = await indexer.sync(collection, ITEMS)
        assert len(report.added) == 3
        assert (await indexer.sync(collection, ITEMS)).unchanged == 3

        results = await store.search(collection, embeddings.embed_query("download invoice"), 2)
        assert results[0][0].question == "Download invoices"
        assert results[0][1] > results[1][1]

        await indexer.sync(collection, ITEMS[:1], prune=True)
        assert list(await store.get_content_hashes(collection)) == ["t-0"]

        with pytest.raises(EmbeddingModelMismatchError):
            await store.ensure_collection(collection, "another-model", DIMENSIONS)
    finally:
        await store.delete_collection(collection)
    with pytest.raises(CollectionNotFoundError):
        await store.search(collection, [0.0] * DIMENSIONS, 1)
