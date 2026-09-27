"""Composition root: wires concrete implementations together from settings.

Keeping construction in one place means the rest of the code only sees interfaces, and tests
(or new deployments) can substitute any dependency by passing it in explicitly.
"""

import logging
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from faq_assistant.agents.responders import ComplianceAgent, LLMAnswerAgent, LocalAnswerAgent
from faq_assistant.config import Settings
from faq_assistant.domain.models import Route
from faq_assistant.guardrails.input_guard import InputGuard
from faq_assistant.guardrails.output_guard import OutputGuard
from faq_assistant.knowledge_base.indexer import EmbeddingIndexer, SyncReport
from faq_assistant.knowledge_base.loader import curate_entries, load_knowledge_base
from faq_assistant.llm.chains import LLMChains, build_chains
from faq_assistant.llm.factory import build_chat_model, build_embeddings
from faq_assistant.retrieval.search import HybridSearcher
from faq_assistant.routing.router import SemanticRouter, default_rules
from faq_assistant.services.assistant import FAQAssistant
from faq_assistant.storage.base import VectorStore
from faq_assistant.storage.memory import InMemoryVectorStore
from faq_assistant.storage.postgres import PostgresVectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Container:
    """Long-lived application dependencies."""

    settings: Settings
    store: VectorStore
    embeddings: Embeddings
    indexer: EmbeddingIndexer
    searcher: HybridSearcher
    assistant: FAQAssistant

    async def aclose(self) -> None:
        """Release resources held by dependencies.

        Returns:
            None. Resources are released as a side effect.
        """
        await self.store.close()


async def create_store(settings: Settings) -> VectorStore:
    """Instantiate and initialise the configured vector store.

    Args:
        settings: Application configuration containing vector store type and credentials.

    Returns:
        An initialized VectorStore instance (in-memory or PostgreSQL).
    """
    if settings.vector_store == "memory":
        return InMemoryVectorStore()
    store = PostgresVectorStore(settings.database_url, settings.embedding_dimensions)
    await store.initialize()
    return store


def create_indexer(
    settings: Settings, store: VectorStore, embeddings: Embeddings
) -> EmbeddingIndexer:
    """Build the embedding indexer for the configured model.

    Args:
        settings: Application configuration containing embedding model and provider.
        store: Vector store instance to sync embeddings with.
        embeddings: Embeddings model for generating vectors.

    Returns:
        An EmbeddingIndexer instance ready for syncing items.
    """
    return EmbeddingIndexer(
        store,
        embeddings,
        embedding_model=f"{settings.embedding_provider}:{settings.embedding_model}",
        dimensions=settings.embedding_dimensions,
    )


async def sync_knowledge_base(
    settings: Settings, indexer: EmbeddingIndexer, *, prune: bool = False
) -> SyncReport:
    """Curate the configured knowledge-base file and sync it into the default collection.

    Args:
        settings: Application configuration containing knowledge base path and collection name.
        indexer: Embedding indexer for syncing items to the vector store.
        prune: Whether to delete items not in the source file.

    Returns:
        A SyncReport summarizing the number of items added, updated, and deleted.
    """
    source = load_knowledge_base(settings.knowledge_base_path)
    curated = curate_entries(source.knowledge_base_items, settings.default_collection)
    return await indexer.sync(settings.default_collection, curated.items, prune=prune)


async def create_container(
    settings: Settings,
    *,
    store: VectorStore | None = None,
    embeddings: Embeddings | None = None,
    chains: LLMChains | None = None,
) -> Container:
    """Build all dependencies. Any argument overrides the settings-driven default.

    Args:
        settings: Application configuration.
        store: Vector store instance; created from settings if None.
        embeddings: Embeddings model; created from settings if None.
        chains: LLM chains; created from settings if None.

    Returns:
        A Container holding all application dependencies.
    """
    store = store or await create_store(settings)
    embeddings = embeddings or build_embeddings(settings)
    chains = chains or build_chains(build_chat_model(settings))

    output_guard = OutputGuard(max_chars=settings.max_answer_chars)
    searcher = HybridSearcher(
        store,
        embeddings,
        top_k=settings.retrieval_top_k,
        lexical_weight=settings.lexical_boost_weight,
    )
    router = SemanticRouter(
        default_rules(
            input_guard=InputGuard(max_chars=settings.max_question_chars),
            router_chain=chains.router,
            accept_threshold=settings.accept_threshold,
            candidate_threshold=settings.candidate_threshold,
            fallback_accept_threshold=settings.fallback_accept_threshold,
        )
    )
    assistant = FAQAssistant(
        searcher,
        router,
        {
            Route.LOCAL: LocalAnswerAgent(
                chains.personalize if settings.personalize_answers else None, output_guard
            ),
            Route.LLM: LLMAnswerAgent(chains.answer, output_guard),
            Route.COMPLIANCE: ComplianceAgent(),
        },
        collection=settings.default_collection,
    )
    container = Container(
        settings=settings,
        store=store,
        embeddings=embeddings,
        indexer=create_indexer(settings, store, embeddings),
        searcher=searcher,
        assistant=assistant,
    )
    # The in-memory store starts empty, so it is always populated; for Postgres this is opt-in
    # (normally done by `faq-admin sync`), and cheap thanks to content hashing.
    if settings.vector_store == "memory" or settings.sync_on_startup:
        try:
            await sync_knowledge_base(settings, container.indexer)
        except BaseException:
            await container.aclose()
            raise
    logger.info("Router rules: %s", router.rule_names)
    return container
