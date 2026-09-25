"""Mapping of domain exceptions to HTTP responses.

Error bodies never include internal details (stack traces, provider messages); those go to
the logs, correlated through the ``X-Request-ID`` header.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from faq_assistant.api.schemas import ErrorResponse
from faq_assistant.domain.errors import (
    CollectionNotFoundError,
    FAQAssistantError,
    KnowledgeBaseError,
    UpstreamRateLimitError,
    UpstreamServiceError,
    VectorStoreError,
)
from faq_assistant.logging_config import request_id_var

logger = logging.getLogger(__name__)

# Most specific first.
_ERROR_MAP: tuple[tuple[type[FAQAssistantError], int, str], ...] = (
    (
        UpstreamRateLimitError,
        status.HTTP_429_TOO_MANY_REQUESTS,
        "The language model provider is rate limiting requests. Please retry shortly.",
    ),
    (
        UpstreamServiceError,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The language model provider is currently unavailable. Please retry later.",
    ),
    (
        CollectionNotFoundError,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The knowledge base has not been indexed yet.",
    ),
    (
        VectorStoreError,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The knowledge base is currently unavailable. Please retry later.",
    ),
    (KnowledgeBaseError, status.HTTP_422_UNPROCESSABLE_CONTENT, "Invalid knowledge base data."),
)


def _error_response(status_code: int, detail: str) -> JSONResponse:
    body = ErrorResponse(detail=detail, request_id=request_id_var.get())
    headers = {"Retry-After": "10"} if status_code in {429, 503} else None
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=headers)


async def _domain_error_handler(request: Request, exc: Exception) -> JSONResponse:
    for error_type, status_code, detail in _ERROR_MAP:
        if isinstance(exc, error_type):
            logger.warning("%s on %s: %s", type(exc).__name__, request.url.path, exc)
            return _error_response(status_code, detail)
    return await _unhandled_error_handler(request, exc)


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error on %s", request.url.path, exc_info=exc)
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal server error.")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach exception handlers to ``app``."""
    app.add_exception_handler(FAQAssistantError, _domain_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
