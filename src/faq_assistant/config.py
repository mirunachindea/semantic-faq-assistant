"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be overridden with an environment variable of the same name
    (case-insensitive), e.g. ``CHAT_MODEL=gpt-4.1-mini``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application -------------------------------------------------------
    app_name: str = "Semantic FAQ Assistant"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # --- Authentication ----------------------------------------------------
    api_tokens: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)
    admin_tokens: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)

    # --- LLM / embeddings (provider-agnostic, resolved through LangChain) --
    llm_provider: str = "openai"
    chat_model: str = "gpt-5.4-mini"
    chat_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)
    llm_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_timeout_seconds: float = Field(default=20.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    max_answer_chars: int = Field(default=2000, gt=0)

    # --- Knowledge base / vector store --------------------------------------
    vector_store: Literal["postgres", "memory"] = "postgres"
    database_url: str = "postgresql://faq:faq@localhost:5432/faq"
    default_collection: str = "faq"
    knowledge_base_path: Path = Path("data/knowledge_base.json")
    sync_on_startup: bool = False

    # --- Retrieval & routing -------------------------------------------------
    retrieval_top_k: int = Field(default=5, gt=0, le=50)
    lexical_boost_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    accept_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    candidate_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    fallback_accept_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    personalize_answers: bool = True
    max_question_chars: int = Field(default=500, gt=0, le=4000)

    # --- Async processing ----------------------------------------------------
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    @field_validator("api_tokens", "admin_tokens", mode="before")
    @classmethod
    def _split_tokens(cls, value: object) -> object:
        """Accept a comma-separated string (the natural env-var format).

        Args:
            value: The raw value from the environment, either a string or already-parsed list.

        Returns:
            A list of token strings parsed from the input.
        """
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        return value

    @model_validator(mode="after")
    def _check_thresholds(self) -> Self:
        """Validate that routing thresholds are in the correct order.

        Args:
            self: The model instance to validate.

        Returns:
            The validated settings instance.
        """
        if not self.candidate_threshold <= self.fallback_accept_threshold <= self.accept_threshold:
            msg = (
                "Expected candidate_threshold <= fallback_accept_threshold <= "
                "accept_threshold, got "
                f"{self.candidate_threshold} / {self.fallback_accept_threshold} / "
                f"{self.accept_threshold}"
            )
            raise ValueError(msg)
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Returns:
        The cached Settings instance (created once and reused).
    """
    return Settings()
