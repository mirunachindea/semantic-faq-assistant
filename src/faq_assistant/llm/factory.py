"""Provider-agnostic model construction.

Everything downstream depends on LangChain's ``BaseChatModel`` / ``Embeddings`` interfaces.
Switching provider or model is a configuration change (``LLM_PROVIDER``, ``CHAT_MODEL``,
``EMBEDDING_PROVIDER``, ``EMBEDDING_MODEL``) as long as the matching ``langchain-<provider>``
integration package is installed.
"""

from typing import Any

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from faq_assistant.config import Settings

# Embedding models that support server-side dimensionality reduction.
_DIMENSION_AWARE_PREFIXES = ("text-embedding-3",)


def _credentials(settings: Settings) -> dict[str, Any]:
    if settings.llm_api_key is None:
        return {}
    return {"api_key": settings.llm_api_key.get_secret_value()}


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model configured in ``settings``."""
    model: BaseChatModel = init_chat_model(
        settings.chat_model,
        model_provider=settings.llm_provider,
        temperature=settings.chat_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        **_credentials(settings),
    )
    return model


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model configured in ``settings``."""
    kwargs: dict[str, Any] = {
        **_credentials(settings),
        "max_retries": settings.llm_max_retries,
        "request_timeout": settings.llm_timeout_seconds,
    }
    if settings.embedding_model.startswith(_DIMENSION_AWARE_PREFIXES):
        kwargs["dimensions"] = settings.embedding_dimensions
    return init_embeddings(settings.embedding_model, provider=settings.embedding_provider, **kwargs)
