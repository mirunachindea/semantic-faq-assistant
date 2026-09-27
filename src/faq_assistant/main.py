"""FastAPI application factory."""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from faq_assistant import __version__
from faq_assistant.api.errors import register_exception_handlers
from faq_assistant.api.routers import admin, health, questions
from faq_assistant.config import Settings, get_settings
from faq_assistant.container import Container, create_container
from faq_assistant.logging_config import configure_logging, request_id_var

logger = logging.getLogger(__name__)

ContainerFactory = Callable[[Settings], Awaitable[Container]]


def create_app(
    settings: Settings | None = None,
    container_factory: ContainerFactory = create_container,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration; defaults to environment-derived settings.
        container_factory: Builds dependencies at startup (overridable in tests).

    Returns:
        A fully configured FastAPI application instance.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    if not settings.api_tokens and not settings.admin_tokens:
        logger.warning("No API_TOKENS configured: every authenticated request will be rejected")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Create and attach the dependency container to ``app``.

        Args:
            app: The FastAPI application instance.

        Yields:
            None. The container is attached to ``app.state.container`` as a side effect.
        """
        app.state.container = await container_factory(settings)
        logger.info("%s started (%s)", settings.app_name, settings.environment)
        try:
            yield
        finally:
            await app.state.container.aclose()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Answers user questions from a curated FAQ knowledge base using hybrid semantic "
            "search, an agentic semantic router and an LLM fallback with guardrails."
        ),
        lifespan=lifespan,
        docs_url=None if settings.environment == "prod" else "/docs",
        redoc_url=None,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Inject a request ID into the context and log request duration.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler to call.

        Returns:
            The HTTP response from the next handler, with the request ID header added.
        """
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(request_id[:64])
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("%s %s %.1fms", request.method, request.url.path, elapsed_ms)
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id[:64]
        return response

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(questions.router)
    app.include_router(admin.router)
    return app
