"""Text normalisation and data-quality rules for knowledge-base entries.

The raw FAQ contains realistic "gotchas": emoji noise, placeholder questions (``"x"``),
answers that are actually user complaints, non-ASCII hyphens and answers that recommend
literal example passwords. Every rule here is small, pure and individually tested so new
rules can be added without touching the ingestion pipeline.
"""

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

# Non-ASCII dashes/spaces that break lexical matching and look identical to users.
_CHAR_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f900-\U0001f9ff️\u200d]+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,])\1+")
_INLINE_WS_RE = re.compile(r"[ \t]+")
_ANY_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Sentences recommending a concrete password: bad security advice that we strip on ingestion.
_LITERAL_PASSWORD_SENTENCE_RE = re.compile(
    r"[^.!?\n]*\bpassword\s+(?:like|such\s+as|e\.g\.)\s*[\"'][^\"']+[\"'][^.!?\n]*[.!?]?",
    re.IGNORECASE,
)
# Answers that read like a user's plea rather than a support answer.
_USER_PLEA_RE = re.compile(r"\b(pls|plz|help me|asap)\b", re.IGNORECASE)

MIN_QUESTION_WORDS = 2
MIN_ANSWER_CHARS = 15


def normalize_text(text: str, *, keep_newlines: bool = False) -> str:
    """Canonicalise unicode, strip control/zero-width characters and collapse whitespace.

    Args:
        text: The text to normalize.
        keep_newlines: Whether to preserve newline characters.

    Returns:
        The normalized text with unicode canonicalized and whitespace collapsed.
    """
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_TRANSLATION)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    if keep_newlines:
        lines = (_INLINE_WS_RE.sub(" ", line).strip() for line in text.splitlines())
        return "\n".join(line for line in lines if line)
    return _ANY_WS_RE.sub(" ", text).strip()


def to_search_text(text: str) -> str:
    """Produce the representation that is embedded / tokenised for search.

    Applied symmetrically to FAQ questions at indexing time and to user queries at query
    time, so both sides live in the same text distribution.

    Args:
        text: The text to prepare for embedding and searching.

    Returns:
        The normalized and cleaned text ready for embedding.
    """
    text = normalize_text(text)
    text = _EMOJI_RE.sub(" ", text)
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    return _ANY_WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens.

    Args:
        text: The text to tokenize.

    Returns:
        A list of lowercase alphanumeric tokens extracted from the text.
    """
    return _WORD_RE.findall(text.lower())


def normalize_category(category: str) -> str:
    """Normalise categories to ``snake_case`` so filters and metadata stay consistent.

    Args:
        category: The category name to normalize.

    Returns:
        The normalized category in snake_case format, or "uncategorized" if empty.
    """
    return "_".join(tokenize(normalize_text(category))) or "uncategorized"


class Severity(StrEnum):
    """How a data-quality issue is handled."""

    WARNING = "warning"  # item is kept (possibly after an automatic fix)
    ERROR = "error"  # item is excluded from the index


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """A data-quality finding attached to a knowledge-base entry."""

    code: str
    severity: Severity
    message: str


@dataclass(frozen=True, slots=True)
class CleanedEntry:
    """Result of running the cleaning pipeline on a single raw entry."""

    question: str
    search_question: str
    answer: str
    category: str
    issues: tuple[QualityIssue, ...]

    @property
    def accepted(self) -> bool:
        """Whether the entry can be indexed."""
        return all(issue.severity is not Severity.ERROR for issue in self.issues)


# A rule inspects (question, search_question, answer) and may return a fixed answer + issue.
RuleResult = tuple[str, QualityIssue] | None
QualityRule = Callable[[str, str, str], RuleResult]


def rule_question_too_short(_question: str, search_question: str, answer: str) -> RuleResult:
    """Reject placeholder questions such as ``"x"`` that would match anything and nothing.

    Returns:
        A tuple of (answer, QualityIssue) if the question is too short, None otherwise.
    """
    words = [token for token in tokenize(search_question) if len(token) > 1]
    if len(words) < MIN_QUESTION_WORDS:
        return answer, QualityIssue(
            "question_too_short",
            Severity.ERROR,
            f"Question has fewer than {MIN_QUESTION_WORDS} meaningful words.",
        )
    return None


def rule_answer_not_informative(_question: str, _search_question: str, answer: str) -> RuleResult:
    """Reject answers that are empty, too short or read like a user complaint.

    Returns:
        A tuple of (answer, QualityIssue) if the answer is not informative, None otherwise.
    """
    stripped = to_search_text(answer)
    if len(stripped) < MIN_ANSWER_CHARS or _USER_PLEA_RE.search(stripped):
        return answer, QualityIssue(
            "answer_not_informative",
            Severity.ERROR,
            "Answer does not contain actionable support content.",
        )
    return None


def rule_noisy_question(question: str, search_question: str, answer: str) -> RuleResult:
    """Flag questions that required emoji/punctuation cleanup before embedding.

    Returns:
        A tuple of (answer, QualityIssue) if cleanup was needed, None otherwise.
    """
    if search_question != normalize_text(question):
        return answer, QualityIssue(
            "noisy_question", Severity.WARNING, "Emoji/repeated punctuation removed for search."
        )
    return None


def rule_literal_password_example(_question: str, _search: str, answer: str) -> RuleResult:
    """Strip sentences that recommend concrete example passwords (unsafe advice).

    Returns:
        A tuple of (fixed_answer, QualityIssue) if sentences were removed, None otherwise.
    """
    fixed = _LITERAL_PASSWORD_SENTENCE_RE.sub("", answer)
    if fixed != answer:
        fixed = normalize_text(fixed, keep_newlines=True)
        return fixed, QualityIssue(
            "literal_password_example",
            Severity.WARNING,
            "Removed a sentence recommending a literal example password.",
        )
    return None


DEFAULT_RULES: tuple[QualityRule, ...] = (
    rule_question_too_short,
    rule_answer_not_informative,
    rule_noisy_question,
    rule_literal_password_example,
)


def clean_entry(
    question: str,
    answer: str,
    category: str,
    rules: tuple[QualityRule, ...] = DEFAULT_RULES,
) -> CleanedEntry:
    """Normalise a raw entry and run all quality rules over it.

    Args:
        question: The raw FAQ question text.
        answer: The raw FAQ answer text.
        category: The raw category name.
        rules: Quality rules to apply; defaults to standard rules.

    Returns:
        A CleanedEntry with normalized text and quality issues found.
    """
    clean_question = normalize_text(question)
    search_question = to_search_text(question)
    clean_answer = normalize_text(answer, keep_newlines=True)
    issues: list[QualityIssue] = []
    for rule in rules:
        result = rule(clean_question, search_question, clean_answer)
        if result is not None:
            clean_answer, issue = result
            issues.append(issue)
    return CleanedEntry(
        question=clean_question,
        search_question=search_question,
        answer=clean_answer,
        category=normalize_category(category),
        issues=tuple(issues),
    )
