"""Domain-level exceptions. The API layer maps them to HTTP responses."""


class FAQAssistantError(Exception):
    """Base class for all application errors."""


class UpstreamServiceError(FAQAssistantError):
    """An external model provider (chat or embeddings) failed or is unavailable."""


class UpstreamRateLimitError(UpstreamServiceError):
    """The model provider rejected the call because of rate limiting / quota."""


class VectorStoreError(FAQAssistantError):
    """The vector store failed or is misconfigured."""


class CollectionNotFoundError(VectorStoreError):
    """The requested collection does not exist."""


class EmbeddingModelMismatchError(VectorStoreError):
    """A collection was embedded with a different model than the one configured."""


class KnowledgeBaseError(FAQAssistantError):
    """The knowledge base source file is missing or malformed."""
