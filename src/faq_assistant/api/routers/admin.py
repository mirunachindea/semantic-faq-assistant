"""Admin endpoints for knowledge-base management (admin token required)."""

from fastapi import APIRouter, HTTPException, Request, status

from faq_assistant.api.dependencies import AdminDep, ContainerDep
from faq_assistant.api.schemas import (
    CollectionSummary,
    IngestRequest,
    TaskAccepted,
    TaskStatus,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/collections", response_model=list[CollectionSummary])
async def list_collections(container: ContainerDep, _admin: AdminDep) -> list[CollectionSummary]:
    """List stored collections and their item counts."""
    return [
        CollectionSummary(
            name=c.name,
            embedding_model=c.embedding_model,
            embedding_dimensions=c.embedding_dimensions,
            item_count=c.item_count,
        )
        for c in await container.store.list_collections()
    ]


@router.post(
    "/collections/{collection}/items",
    response_model=TaskAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_items(
    collection: str,
    payload: IngestRequest,
    request: Request,
    container: ContainerDep,
    _admin: AdminDep,
) -> TaskAccepted:
    """Queue curation + incremental embedding of items into ``collection`` (Celery)."""
    if container.settings.vector_store != "postgres":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Asynchronous ingestion requires the postgres vector store.",
        )
    # Imported lazily so the API can run without Celery configured (e.g. memory mode, tests).
    from faq_assistant.worker.tasks import sync_collection_task  # noqa: PLC0415

    task = sync_collection_task.delay(
        collection,
        [item.model_dump() for item in payload.knowledge_base_items],
        prune=payload.prune,
    )
    return TaskAccepted(
        task_id=task.id,
        status_url=str(request.url_for("get_task_status", task_id=task.id)),
    )


@router.get("/tasks/{task_id}", response_model=TaskStatus, name="get_task_status")
async def get_task_status(task_id: str, _admin: AdminDep) -> TaskStatus:
    """State of a queued ingestion job."""
    from faq_assistant.worker.celery_app import celery_app  # noqa: PLC0415

    result = celery_app.AsyncResult(task_id)
    if result.failed():
        return TaskStatus(task_id=task_id, state=result.state, error=str(result.result))
    payload = result.result if result.successful() else None
    return TaskStatus(task_id=task_id, state=result.state, result=payload)
