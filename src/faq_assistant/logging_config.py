"""Logging setup with request-id correlation."""

import logging
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once; safe to call repeatedly."""
    root = logging.getLogger()
    if any(getattr(handler, "_faq_handler", False) for handler in root.handlers):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler()
    handler._faq_handler = True  # type: ignore[attr-defined]
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Third-party clients are chatty at INFO and may log request payloads.
    for noisy in ("httpx", "openai", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
