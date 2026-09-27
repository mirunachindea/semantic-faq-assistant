"""Request/response models of the public HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from faq_assistant.domain.models import AnswerSource, AssistantAnswer
from faq_assistant.knowledge_base.loader import RawFAQEntry

NOT_APPLICABLE = "N/A"


class AskQuestionRequest(BaseModel):
    """Body of ``POST /ask-question``."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"user_question": "How do I reset my account?"}]},
    )

    user_question: str = Field(min_length=1, max_length=2000)

    @field_validator("user_question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Ensure user_question is not blank.

        Args:
            value: The user_question value to validate.

        Returns:
            The stripped user_question string.
        """
        if not value.strip():
            msg = "user_question must not be blank"
            raise ValueError(msg)
        return value.strip()


class AskQuestionResponse(BaseModel):
    """Answer returned by ``POST /ask-question``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source": "local",
                    "matched_question": "How can I restore my account to its default settings?",
                    "answer": "Go to settings and click on 'Restore Default'.",
                    "category": "settings",
                    "similarity_score": 0.83,
                }
            ]
        }
    )

    source: AnswerSource
    matched_question: str = NOT_APPLICABLE
    answer: str
    category: str | None = None
    similarity_score: float | None = None

    @classmethod
    def from_domain(cls, answer: AssistantAnswer) -> "AskQuestionResponse":
        """Map the domain answer onto the public contract.

        Args:
            answer: The domain answer model to convert.

        Returns:
            An AskQuestionResponse with the mapped answer data.
        """
        return cls(
            source=answer.source,
            matched_question=answer.matched_question or NOT_APPLICABLE,
            answer=answer.answer,
            category=answer.category,
            similarity_score=answer.similarity_score,
        )


class ErrorResponse(BaseModel):
    """Uniform error payload."""

    detail: str
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Health/readiness payload."""

    status: Literal["ok", "degraded"]
    vector_store: bool | None = None


class IngestRequest(BaseModel):
    """Body of the admin ingestion endpoint (same shape as the KB source file)."""

    knowledge_base_items: list[RawFAQEntry] = Field(min_length=1, max_length=5000)
    prune: bool = False


class TaskAccepted(BaseModel):
    """Returned when an asynchronous job has been queued."""

    task_id: str
    status_url: str


class TaskStatus(BaseModel):
    """State of an asynchronous job."""

    task_id: str
    state: str
    result: dict[str, object] | None = None
    error: str | None = None


class CollectionSummary(BaseModel):
    """A stored collection."""

    name: str
    embedding_model: str
    embedding_dimensions: int
    item_count: int
