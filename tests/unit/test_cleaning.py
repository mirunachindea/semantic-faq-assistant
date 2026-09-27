import pytest

from faq_assistant.knowledge_base.cleaning import (
    Severity,
    clean_entry,
    normalize_category,
    normalize_text,
    to_search_text,
)
from faq_assistant.knowledge_base.loader import RawFAQEntry, curate_entries, load_knowledge_base
from tests.conftest import KB_PATH


def test_normalize_text_replaces_unicode_dashes_and_zero_width() -> None:
    """Test text normalization handles unicode dashes and zero-width characters."""
    assert normalize_text("Two\u2011Factor\u200b  code") == "Two-Factor code"


def test_normalize_text_keeps_newlines_when_requested() -> None:
    """Test text normalization preserves newlines when keep_newlines is True."""
    text = "Contact support:\n-  Date of loss\n\n- Type"
    assert normalize_text(text, keep_newlines=True) == "Contact support:\n- Date of loss\n- Type"


def test_search_text_strips_emoji_and_repeated_punctuation() -> None:
    """Test search text cleaning removes emoji and repeated punctuation."""
    assert to_search_text("help!!! 😭😭😭 my account is locked") == "help! my account is locked"


@pytest.mark.parametrize(
    ("raw", "expected"), [("Data Recovery", "data_recovery"), ("  ", "uncategorized")]
)
def test_normalize_category(raw: str, expected: str) -> None:
    """Test category normalization converts to snake_case and handles empty input.

    Args:
        raw: Raw category string.
        expected: Expected normalized category.
    """
    assert normalize_category(raw) == expected


def test_placeholder_question_is_rejected() -> None:
    """Test placeholder questions are rejected as too short."""
    entry = clean_entry("x", "Please provide more details about your issue.", "troubleshooting")
    assert not entry.accepted
    assert "question_too_short" in {i.code for i in entry.issues}


def test_user_plea_answer_is_rejected() -> None:
    """Test non-informative answers are rejected."""
    entry = clean_entry("help!!! 😭 my account is locked", "pls help me unlock it ASAP!!! 🔓", "x")
    assert not entry.accepted
    assert "answer_not_informative" in {i.code for i in entry.issues}


def test_literal_password_example_is_removed_but_entry_kept() -> None:
    """Test literal password examples are removed from answers."""
    entry = clean_entry(
        "What do I do if my account has been compromised?",
        "Immediately reset your password and contact our security team. "
        "Use a password like \"Secure!Pass123\" or 'Strong_Pass' (avoid common patterns).",
        "security_incident",
    )
    assert entry.accepted
    assert entry.answer == "Immediately reset your password and contact our security team."
    assert [i.code for i in entry.issues] == ["literal_password_example"]
    assert entry.issues[0].severity is Severity.WARNING


def test_short_keyword_questions_are_kept() -> None:
    """Test short keyword-style questions are accepted."""
    assert clean_entry("Edit avatar?", "Open Profile and upload a new image.", "profile").accepted


def test_curation_of_provided_knowledge_base() -> None:
    """Test knowledge base curation filters and normalizes entries."""
    source = load_knowledge_base(KB_PATH)
    report = curate_entries(source.knowledge_base_items, "faq")
    rejected = {r.question for r in report.rejected}
    assert rejected == {"x", "help!!! 😭😭😭 my account is locked"}
    assert len(report.items) == len(source.knowledge_base_items) - 2
    assert len({item.id for item in report.items}) == len(report.items)
    assert all("\u2011" not in item.answer for item in report.items)


def test_duplicate_questions_are_rejected() -> None:
    """Test duplicate questions are detected and rejected."""
    entries = [
        RawFAQEntry(question="Cancel subscription", answer="Settings -> Subscription -> Cancel."),
        RawFAQEntry(question="cancel  subscription", answer="Another, conflicting answer."),
    ]
    report = curate_entries(entries, "faq")
    assert len(report.items) == 1
    assert report.rejected[0].issues[-1].code == "duplicate_question"


def test_ids_are_stable_and_collection_scoped() -> None:
    """Test generated IDs are stable but scoped to collection."""
    entry = [RawFAQEntry(question="Upgrade my plan", answer="Settings -> Subscription.")]
    first = curate_entries(entry, "faq").items[0].id
    assert first == curate_entries(entry, "faq").items[0].id
    assert first != curate_entries(entry, "billing").items[0].id
