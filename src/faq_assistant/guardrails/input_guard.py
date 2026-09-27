"""Deterministic first line of defence against prompt injection and abusive input.

This layer is cheap (no model call) and catches the well-known attack families. It is
intentionally *not* the only defence: the LLM router also classifies manipulation attempts,
prompts are hardened, and outputs are checked by :mod:`faq_assistant.guardrails.output_guard`.
"""

import re
from dataclasses import dataclass

from faq_assistant.knowledge_base.cleaning import normalize_text

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for name, pattern in (
        (
            "override_instructions",
            r"\b(ignore|disregard|forget|override|bypass|skip)\s+(all\s+|any\s+|of\s+)?"
            r"(the\s+|your\s+|these\s+|those\s+|my\s+)?"
            r"(previous|prior|above|earlier|preceding|system|original|initial|safety)?\s*"
            r"(instructions?|rules|prompts?|guidelines|directives|guardrails|restrictions)\b",
        ),
        (
            "prompt_extraction",
            r"\b(reveal|show|print|repeat|output|display|tell|give|leak|what\s+(is|are))\b"
            r".{0,40}\b(system|hidden|initial|original|developer|secret)\s+"
            r"(prompt|instructions?|message|rules?)\b",
        ),
        (
            "role_hijack",
            r"\b(you\s+are\s+now|from\s+now\s+on\s+you|pretend\s+(to\s+be|you\s+are)|"
            r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered|evil|different|new)|roleplay\s+as|"
            r"new\s+persona)\b",
        ),
        (
            "jailbreak_keywords",
            r"\b(jailbreak|jail\s*broken|DAN\s+mode|do\s+anything\s+now|developer\s+mode|"
            r"god\s+mode|no\s+restrictions|without\s+(any\s+)?(restrictions|filters|limits))\b",
        ),
        (
            "chat_template_tokens",
            r"(<\|[a-z_]+\|>|\[/?INST\]|<</?SYS>>|^\s*#{2,}\s*(system|instruction)|"
            r"^\s*(system|assistant)\s*:)",
        ),
        ("encoded_payload", r"[A-Za-z0-9+/]{60,}={0,2}"),
    )
)

_MIN_PRINTABLE_RATIO = 0.8


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """Result of a guardrail check."""

    allowed: bool
    text: str
    reason: str | None = None


class InputGuard:
    """Normalises user input and blocks known injection / abuse patterns."""

    def __init__(self, max_chars: int = 500):
        """Initialize the input guard.

        Args:
            max_chars: Maximum allowed input length in characters.
        """
        self._max_chars = max_chars

    def check(self, raw_text: str) -> GuardVerdict:
        """Return the normalised text and whether it may proceed.

        Args:
            raw_text: The user's input text to validate and normalize.

        Returns:
            A GuardVerdict indicating if the input is allowed and the normalized text.
        """
        # Normalisation defeats trivial obfuscation (zero-width chars, homoglyph dashes,
        # full-width letters via NFKC) before pattern matching.
        text = normalize_text(raw_text)
        if not text:
            return GuardVerdict(allowed=False, text=text, reason="empty_input")
        if len(text) > self._max_chars:
            return GuardVerdict(allowed=False, text=text, reason="input_too_long")
        printable = sum(ch.isprintable() for ch in text) / len(text)
        if printable < _MIN_PRINTABLE_RATIO:
            return GuardVerdict(allowed=False, text=text, reason="non_printable_input")
        multiline = normalize_text(raw_text, keep_newlines=True)
        for name, pattern in _INJECTION_PATTERNS:
            if pattern.search(text) or pattern.search(multiline):
                return GuardVerdict(allowed=False, text=text, reason=f"injection:{name}")
        return GuardVerdict(allowed=True, text=text)
