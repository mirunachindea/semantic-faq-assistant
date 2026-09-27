import pytest

from faq_assistant.domain.errors import EmbeddingModelMismatchError, UpstreamServiceError
from faq_assistant.domain.models import FAQItem
from faq_assistant.knowledge_base.indexer import EmbeddingIndexer
from faq_assistant.retrieval.search import HybridSearcher, combine_scores, lexical_overlap
from faq_assistant.storage.memory import InMemoryVectorStore
from tests.fakes import DIMENSIONS, BagOfWordsEmbeddings


def _item(item_id: str, question: str, answer: str = "Some helpful answer.") -> FAQItem:
    """Create a test FAQ item.

    Args:
        item_id: Unique item identifier.
        question: Question text.
        answer: Answer text (default provided).

    Returns:
        FAQItem: Test FAQ item.
    """
    return FAQItem(id=item_id, question=question, answer=answer, category="general")


ITEMS = [
    _item("1", "How do I cancel my subscription?", "Settings -> Subscription -> Cancel."),
    _item("2", "Where can I download invoices?", "Billing -> Invoices."),
    _item("3", "Enable two-factor authentication", "Settings -> Security -> Two-Factor."),
]


@pytest.fixture
def store() -> InMemoryVectorStore:
    """In-memory vector store for testing.

    Returns:
        InMemoryVectorStore: Test vector store instance.
    """
    return InMemoryVectorStore()


@pytest.fixture
def indexer(store: InMemoryVectorStore, embeddings: BagOfWordsEmbeddings) -> EmbeddingIndexer:
    """Embedding indexer with test store and embeddings.

    Args:
        store: In-memory vector store fixture.
        embeddings: Test embeddings fixture.

    Returns:
        EmbeddingIndexer: Indexer instance for testing.
    """
    return EmbeddingIndexer(store, embeddings, embedding_model="fake", dimensions=DIMENSIONS)


async def test_first_sync_embeds_question_and_document_per_item(
    indexer: EmbeddingIndexer, embeddings: BagOfWordsEmbeddings
) -> None:
    """Test first sync embeds both question and answer for each item.

    Args:
        indexer: Embedding indexer fixture.
        embeddings: Test embeddings fixture.
    """
    report = await indexer.sync("faq", ITEMS)
    assert len(report.added) == 3
    assert report.embedded_texts == 6
    assert len(embeddings.embedded_texts) == 6


async def test_resync_is_token_free_and_only_changed_items_are_reembedded(
    indexer: EmbeddingIndexer, embeddings: BagOfWordsEmbeddings
) -> None:
    """Test resync avoids re-embedding unchanged items.

    Args:
        indexer: Embedding indexer fixture.
        embeddings: Test embeddings fixture.
    """
    await indexer.sync("faq", ITEMS)
    embeddings.embedded_texts.clear()

    unchanged = await indexer.sync("faq", ITEMS)
    assert unchanged.unchanged == 3
    assert embeddings.embedded_texts == []

    changed = [*ITEMS[:2], _item("3", ITEMS[2].question, "New answer text.")]
    report = await indexer.sync("faq", changed)
    assert report.updated == ["3"]
    assert report.unchanged == 2
    assert len(embeddings.embedded_texts) == 2


async def test_sync_without_prune_keeps_existing_items(
    indexer: EmbeddingIndexer, store: InMemoryVectorStore
) -> None:
    """Test sync prunes items when requested.

    Args:
        indexer: Embedding indexer fixture.
        store: In-memory vector store fixture.
    """
    await indexer.sync("faq", ITEMS)
    await indexer.sync("faq", ITEMS[:1])
    assert len(await store.get_content_hashes("faq")) == 3

    report = await indexer.sync("faq", ITEMS[:1], prune=True)
    assert sorted(report.deleted) == ["2", "3"]
    assert list(await store.get_content_hashes("faq")) == ["1"]


async def test_collections_are_isolated_and_model_locked(
    indexer: EmbeddingIndexer, store: InMemoryVectorStore, embeddings: BagOfWordsEmbeddings
) -> None:
    """Test collections are isolated and embedding model is locked per collection.

    Args:
        indexer: Embedding indexer fixture.
        store: In-memory vector store fixture.
        embeddings: Test embeddings fixture.
    """
    await indexer.sync("faq", ITEMS)
    await indexer.sync("billing", ITEMS[1:2])
    counts = {c.name: c.item_count for c in await store.list_collections()}
    assert counts == {"billing": 1, "faq": 3}

    other_model = EmbeddingIndexer(
        store, embeddings, embedding_model="other", dimensions=DIMENSIONS
    )
    with pytest.raises(EmbeddingModelMismatchError):
        await other_model.sync("faq", ITEMS)


async def test_embedding_failures_become_upstream_errors(
    indexer: EmbeddingIndexer, embeddings: BagOfWordsEmbeddings
) -> None:
    """Test embedding failures are translated to UpstreamServiceError.

    Args:
        indexer: Embedding indexer fixture.
        embeddings: Test embeddings fixture.
    """
    embeddings.fail_with = RuntimeError("connection reset")
    with pytest.raises(UpstreamServiceError):
        await indexer.sync("faq", ITEMS)


async def test_hybrid_search_ranks_best_match_first(
    indexer: EmbeddingIndexer, store: InMemoryVectorStore, embeddings: BagOfWordsEmbeddings
) -> None:
    """Test hybrid search combines dense and lexical ranking.

    Args:
        indexer: Embedding indexer fixture.
        store: In-memory vector store fixture.
        embeddings: Test embeddings fixture.
    """
    await indexer.sync("faq", ITEMS)
    searcher = HybridSearcher(store, embeddings, top_k=3, lexical_weight=0.2)
    hits = await searcher.search("how can I cancel the subscription??? 😭", "faq")
    assert hits[0].item.id == "1"
    assert hits[0].score >= hits[0].dense_score
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_lexical_overlap_ignores_stopwords_and_plural() -> None:
    """Test lexical overlap calculation handles stopwords and pluralization."""
    assert lexical_overlap("download my invoices", "Where can I download invoice?") == 1.0
    assert lexical_overlap("the and of", "anything") == 0.0


@pytest.mark.parametrize(("dense", "lexical"), [(0.0, 1.0), (0.5, 0.5), (0.9, 1.0), (1.0, 1.0)])
def test_lexical_boost_is_bounded_and_monotonic(dense: float, lexical: float) -> None:
    """Test score combination is bounded and monotonic.

    Args:
        dense: Dense search score.
        lexical: Lexical search score.
    """
    score = combine_scores(dense, lexical, 0.2)
    assert dense <= score <= 1.0
    assert combine_scores(dense, 0.0, 0.2) == dense
