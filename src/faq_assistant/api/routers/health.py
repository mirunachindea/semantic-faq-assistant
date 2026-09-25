"""Liveness and readiness probes."""

from fastapi import APIRouter, Response, status

from faq_assistant.api.dependencies import ContainerDep
from faq_assistant.api.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    """The process is up."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness(container: ContainerDep, response: Response) -> HealthResponse:
    """Dependencies needed to serve traffic are reachable."""
    store_ok = await container.store.ping()
    if not store_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if store_ok else "degraded", vector_store=store_ok)
