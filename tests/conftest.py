from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from faq_assistant.config import Settings
from faq_assistant.container import Container, create_container
from faq_assistant.storage.memory import InMemoryVectorStore
from tests.fakes import DIMENSIONS, BagOfWordsEmbeddings, make_chains

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"
API_TOKEN = "test-token"
ADMIN_TOKEN = "admin-token"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        vector_store="memory",
        knowledge_base_path=KB_PATH,
        embedding_dimensions=DIMENSIONS,
        api_tokens=[API_TOKEN],
        admin_tokens=[ADMIN_TOKEN],
        llm_api_key=None,
    )


@pytest.fixture
def embeddings() -> BagOfWordsEmbeddings:
    return BagOfWordsEmbeddings()


@pytest.fixture
async def container(
    settings: Settings, embeddings: BagOfWordsEmbeddings
) -> AsyncIterator[Container]:
    chains, _ = make_chains()
    built = await create_container(
        settings, store=InMemoryVectorStore(), embeddings=embeddings, chains=chains
    )
    yield built
    await built.aclose()
