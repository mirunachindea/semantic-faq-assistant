"""Translate provider-specific exceptions into domain errors."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from faq_assistant.domain.errors import UpstreamRateLimitError, UpstreamServiceError

logger = logging.getLogger(__name__)

_HTTP_TOO_MANY_REQUESTS = 429


def _is_rate_limit(exc: BaseException) -> bool:
    # Provider-agnostic: OpenAI, Anthropic, Google... all expose either a *RateLimit* error
    # class or an HTTP status code on the exception.
    return (
        "ratelimit" in type(exc).__name__.lower()
        or getattr(exc, "status_code", None) == _HTTP_TOO_MANY_REQUESTS
    )


@asynccontextmanager
async def upstream_call(operation: str) -> AsyncIterator[None]:
    """Wrap a model call, converting any failure into an :class:`UpstreamServiceError`.

    Covers transport errors, timeouts, rate limits, content-filter refusals and malformed
    (unparseable / schema-violating) structured outputs.
    """
    try:
        yield
    except UpstreamServiceError:
        raise
    except Exception as exc:
        if _is_rate_limit(exc):
            logger.warning("%s rate limited by provider", operation)
            msg = f"{operation}: provider rate limit reached"
            raise UpstreamRateLimitError(msg) from exc
        logger.exception("%s failed", operation)
        msg = f"{operation} failed: {type(exc).__name__}"
        raise UpstreamServiceError(msg) from exc
