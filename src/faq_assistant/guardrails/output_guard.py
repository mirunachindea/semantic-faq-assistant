"""Validation of model outputs before they reach the user."""

import re

from faq_assistant.guardrails.input_guard import GuardVerdict
from faq_assistant.knowledge_base.cleaning import normalize_text
from faq_assistant.llm.prompts import PROMPT_CANARY

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-(proj-)?[A-Za-z0-9_\-]{20,}"),  # OpenAI-style keys
    re.compile(r"\bsk_live_[A-Za-z0-9]{16,}"),  # Stripe-style live keys (not the \w{32} hint)
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access keys
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_PROMPT_LEAK_RE = re.compile(
    r"\b(my|the)\s+(system\s+prompt|instructions\s+(say|are|tell))\b|security rules \(highest",
    re.IGNORECASE,
)


class OutputGuard:
    """Rejects empty, leaking or oversized model outputs."""

    def __init__(self, max_chars: int = 2000):
        """Initialize the output guard.

        Args:
            max_chars: Maximum allowed output length in characters.
        """
        self._max_chars = max_chars

    def check(self, raw_text: str | None) -> GuardVerdict:
        """Validate (and lightly normalise) a model output.

        Args:
            raw_text: The model's output text to validate.

        Returns:
            A GuardVerdict indicating if the output is allowed and the normalized text.
        """
        if not isinstance(raw_text, str):
            return GuardVerdict(allowed=False, text="", reason="malformed_output")
        text = normalize_text(raw_text, keep_newlines=True)
        if not text:
            return GuardVerdict(allowed=False, text=text, reason="empty_output")
        if PROMPT_CANARY in text or _PROMPT_LEAK_RE.search(text):
            return GuardVerdict(allowed=False, text=text, reason="prompt_leak")
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            return GuardVerdict(allowed=False, text=text, reason="secret_in_output")
        if len(text) > self._max_chars:
            cut = text[: self._max_chars].rsplit(" ", 1)[0]
            return GuardVerdict(allowed=True, text=f"{cut}…", reason="truncated")
        return GuardVerdict(allowed=True, text=text)
