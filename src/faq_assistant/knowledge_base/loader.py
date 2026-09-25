"""Loading and curating knowledge-base source files."""

import json
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from faq_assistant.domain.errors import KnowledgeBaseError
from faq_assistant.domain.models import FAQItem
from faq_assistant.knowledge_base.cleaning import QualityIssue, Severity, clean_entry

logger = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.UUID("6f1d8a4e-3a57-4c43-9b8e-5f0c2b7d9e11")


class RawFAQEntry(BaseModel):
    """Schema of a single entry in the source JSON."""

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    category: str = "uncategorized"
    id: str | None = Field(default=None, description="Optional stable id; derived if omitted.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseFile(BaseModel):
    """Schema of the knowledge-base source JSON."""

    knowledge_base_items: list[RawFAQEntry]


@dataclass(frozen=True, slots=True)
class RejectedEntry:
    """A source entry excluded from the index, with the reasons why."""

    question: str
    issues: tuple[QualityIssue, ...]


@dataclass(slots=True)
class CurationReport:
    """Outcome of curating a knowledge-base file."""

    items: list[FAQItem] = field(default_factory=list)
    rejected: list[RejectedEntry] = field(default_factory=list)
    warnings: dict[str, list[QualityIssue]] = field(default_factory=dict)


def derive_item_id(collection: str, search_question: str) -> str:
    """Stable id: the same question in the same collection always maps to the same id."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{collection}:{search_question.casefold()}"))


def curate_entries(entries: Iterable[RawFAQEntry], collection: str) -> CurationReport:
    """Clean, validate and de-duplicate raw entries into indexable :class:`FAQItem` objects."""
    report = CurationReport()
    seen_ids: set[str] = set()
    for entry in entries:
        cleaned = clean_entry(entry.question, entry.answer, entry.category)
        issues = list(cleaned.issues)
        item_id = entry.id or derive_item_id(collection, cleaned.search_question)
        if item_id in seen_ids:
            issues.append(
                QualityIssue("duplicate_question", Severity.ERROR, "Duplicate of an earlier entry.")
            )
        if not cleaned.accepted or item_id in seen_ids:
            report.rejected.append(RejectedEntry(cleaned.question, tuple(issues)))
            continue
        seen_ids.add(item_id)
        warnings = [issue for issue in issues if issue.severity is Severity.WARNING]
        if warnings:
            report.warnings[cleaned.question] = warnings
        report.items.append(
            FAQItem(
                id=item_id,
                question=cleaned.question,
                answer=cleaned.answer,
                category=cleaned.category,
                metadata={
                    **entry.metadata,
                    "search_question": cleaned.search_question,
                    "quality_flags": [issue.code for issue in warnings],
                },
            )
        )
    logger.info(
        "Curated collection %r: %d accepted, %d rejected, %d with warnings",
        collection,
        len(report.items),
        len(report.rejected),
        len(report.warnings),
    )
    return report


def parse_knowledge_base(payload: Any) -> KnowledgeBaseFile:
    """Validate an already-decoded knowledge-base payload."""
    try:
        return KnowledgeBaseFile.model_validate(payload)
    except ValidationError as exc:
        msg = f"Invalid knowledge base payload: {exc}"
        raise KnowledgeBaseError(msg) from exc


def load_knowledge_base(path: Path) -> KnowledgeBaseFile:
    """Read and validate a knowledge-base JSON file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        msg = f"Knowledge base file not found: {path}"
        raise KnowledgeBaseError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Knowledge base file is not valid JSON: {path} ({exc})"
        raise KnowledgeBaseError(msg) from exc
    return parse_knowledge_base(payload)
