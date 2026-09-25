"""Core domain models shared across layers."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FAQItem(BaseModel):
    """A curated knowledge-base entry, ready to be embedded and searched."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    answer: str
    category: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def question_text(self) -> str:
        """Text used for the question-only embedding (question-to-question matching)."""
        return str(self.metadata.get("search_question", self.question))

    @property
    def document_text(self) -> str:
        """Text used for the full-document embedding (question-to-content matching)."""
        topic = self.category.replace("_", " ")
        return f"Topic: {topic}\nQuestion: {self.question_text}\nAnswer: {self.answer}"


class SearchHit(BaseModel):
    """A retrieval candidate with its individual and combined scores."""

    model_config = ConfigDict(frozen=True)

    item: FAQItem
    dense_score: float
    lexical_score: float
    score: float


class Route(StrEnum):
    """Destinations the semantic router can send a question to."""

    LOCAL = "local"
    LLM = "llm"
    COMPLIANCE = "compliance"


class AnswerSource(StrEnum):
    """Public ``source`` field of the API response.

    ``openai`` is kept as the wire value for contract compatibility with the challenge spec,
    even though the underlying chat model is configurable.
    """

    LOCAL = "local"
    OPENAI = "openai"
    COMPLIANCE = "compliance"


class RouteDecision(BaseModel):
    """Outcome of the semantic router."""

    model_config = ConfigDict(frozen=True)

    route: Route
    reason: str
    decided_by: str
    hit: SearchHit | None = None


class AssistantAnswer(BaseModel):
    """Final answer produced by the assistant."""

    source: AnswerSource
    answer: str
    matched_question: str | None = None
    category: str | None = None
    similarity_score: float | None = None
    route_reason: str
