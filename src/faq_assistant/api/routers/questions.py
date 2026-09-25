"""Question answering endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from faq_assistant.api.dependencies import AssistantDep, TokenDep, get_settings_dep
from faq_assistant.api.schemas import AskQuestionRequest, AskQuestionResponse, ErrorResponse
from faq_assistant.config import Settings

router = APIRouter(tags=["questions"])


@router.post(
    "/ask-question",
    response_model=AskQuestionResponse,
    summary="Answer a user question",
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid token"},
        429: {"model": ErrorResponse, "description": "Model provider rate limit"},
        503: {"model": ErrorResponse, "description": "Model provider or store unavailable"},
    },
)
async def ask_question(
    payload: AskQuestionRequest,
    assistant: AssistantDep,
    settings: Annotated[Settings, Depends(get_settings_dep)],
    _token: TokenDep,
) -> AskQuestionResponse:
    """Answer from the FAQ knowledge base, fall back to the LLM, or refuse.

    * ``source="local"``: a semantically matching FAQ entry was found (answer personalised).
    * ``source="openai"``: in-scope question not covered by the FAQ, answered by the LLM.
    * ``source="compliance"``: out-of-scope or unsafe question, fixed refusal message.
    """
    if len(payload.user_question) > settings.max_question_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"user_question must be at most {settings.max_question_chars} characters",
        )
    answer = await assistant.ask(payload.user_question)
    return AskQuestionResponse.from_domain(answer)
