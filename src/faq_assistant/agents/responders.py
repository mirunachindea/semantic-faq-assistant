"""Agents that produce the final answer for each route."""

import logging
from typing import Protocol

from langchain_core.runnables import Runnable

from faq_assistant.domain.errors import UpstreamServiceError
from faq_assistant.domain.models import AnswerSource, AssistantAnswer, RouteDecision
from faq_assistant.guardrails.output_guard import OutputGuard
from faq_assistant.llm.errors import upstream_call

logger = logging.getLogger(__name__)

COMPLIANCE_MESSAGE = (
    "This is not really what I was trained for, therefore I cannot answer. Try again."
)
SAFE_FALLBACK_MESSAGE = (
    "Sorry, I couldn't produce a reliable answer to that. "
    "Please rephrase your question or contact our support team."
)


class Responder(Protocol):
    """Produces an answer for a routed question."""

    async def respond(self, question: str, decision: RouteDecision) -> AssistantAnswer:
        """Answer ``question`` according to ``decision``."""
        ...


class ComplianceAgent:
    """Returns the fixed compliance message for out-of-scope or unsafe questions.

    Deliberately deterministic: no model call, so it cannot be talked out of its answer and
    it does not reveal *why* the question was refused (useful to attackers probing guards).
    """

    async def respond(self, question: str, decision: RouteDecision) -> AssistantAnswer:
        """Return the compliance message."""
        return AssistantAnswer(
            source=AnswerSource.COMPLIANCE,
            answer=COMPLIANCE_MESSAGE,
            route_reason=decision.reason,
        )


class LocalAnswerAgent:
    """Answers from the knowledge base, optionally personalised by the LLM.

    The KB answer is the source of truth: if personalisation fails or its output does not
    pass the output guard, the curated answer is returned verbatim.
    """

    def __init__(
        self,
        personalize_chain: Runnable[dict[str, str], str] | None,
        output_guard: OutputGuard,
    ):
        self._personalize = personalize_chain
        self._guard = output_guard

    async def respond(self, question: str, decision: RouteDecision) -> AssistantAnswer:
        """Return the matched FAQ answer, personalised when possible."""
        if decision.hit is None:
            msg = "Local route requires a matched FAQ entry"
            raise ValueError(msg)
        item = decision.hit.item
        answer = item.answer
        if self._personalize is not None:
            answer = await self._personalized(question, item.question, item.answer)
        return AssistantAnswer(
            source=AnswerSource.LOCAL,
            answer=answer,
            matched_question=item.question,
            category=item.category,
            similarity_score=decision.hit.score,
            route_reason=decision.reason,
        )

    async def _personalized(self, question: str, matched_question: str, reference: str) -> str:
        assert self._personalize is not None  # noqa: S101 - narrowed by caller
        try:
            async with upstream_call("answer personalisation"):
                output = await self._personalize.ainvoke(
                    {
                        "question": question,
                        "matched_question": matched_question,
                        "reference_answer": reference,
                    }
                )
        except UpstreamServiceError:
            logger.warning("Personalisation failed; returning curated answer verbatim")
            return reference
        verdict = self._guard.check(output)
        if not verdict.allowed:
            logger.warning(
                "Personalised answer rejected (%s); using curated answer", verdict.reason
            )
            return reference
        return verdict.text


class LLMAnswerAgent:
    """Answers questions that are in scope but not covered by the knowledge base."""

    def __init__(self, answer_chain: Runnable[dict[str, str], str], output_guard: OutputGuard):
        self._chain = answer_chain
        self._guard = output_guard

    async def respond(self, question: str, decision: RouteDecision) -> AssistantAnswer:
        """Generate an answer; provider failures propagate as :class:`UpstreamServiceError`."""
        async with upstream_call("LLM answer"):
            output = await self._chain.ainvoke({"question": question})
        verdict = self._guard.check(output)
        if not verdict.allowed:
            logger.warning("LLM answer rejected by output guard: %s", verdict.reason)
        return AssistantAnswer(
            source=AnswerSource.OPENAI,
            answer=verdict.text if verdict.allowed else SAFE_FALLBACK_MESSAGE,
            route_reason=decision.reason,
        )
