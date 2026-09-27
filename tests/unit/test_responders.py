from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

from faq_assistant.agents.responders import (
    COMPLIANCE_MESSAGE,
    SAFE_FALLBACK_MESSAGE,
    ComplianceAgent,
    LLMAnswerAgent,
    LocalAnswerAgent,
)
from faq_assistant.domain.errors import UpstreamRateLimitError, UpstreamServiceError
from faq_assistant.domain.models import AnswerSource, FAQItem, Route, RouteDecision, SearchHit
from faq_assistant.guardrails.output_guard import OutputGuard
from faq_assistant.llm.prompts import PROMPT_CANARY

ITEM = FAQItem(id="1", question="Cancel subscription", answer="Settings -> Cancel.", category="s")
HIT = SearchHit(item=ITEM, dense_score=0.9, lexical_score=1.0, score=0.92)
LOCAL = RouteDecision(route=Route.LOCAL, hit=HIT, reason="r", decided_by="t")
LLM = RouteDecision(route=Route.LLM, reason="r", decided_by="t")


def _raise(exc: Exception) -> Any:
    """Create a function that raises an exception when called.

    Args:
        exc: Exception to raise.

    Returns:
        Callable that raises the given exception.
    """

    def _fn(_: Any) -> Any:
        raise exc

    return _fn


async def test_local_answer_is_personalised() -> None:
    """Test local answer is personalized through personalizer chain.

    Verifies that LocalAnswerAgent applies personalization to FAQ answers.
    """
    agent = LocalAnswerAgent(
        RunnableLambda(lambda _: "Sure! Go to Settings -> Cancel."), OutputGuard()
    )
    answer = await agent.respond("stop my plan", LOCAL)
    assert answer.source is AnswerSource.LOCAL
    assert answer.answer == "Sure! Go to Settings -> Cancel."
    assert answer.matched_question == "Cancel subscription"
    assert answer.similarity_score == 0.92


@pytest.mark.parametrize(
    "personalize",
    [
        RunnableLambda(_raise(RuntimeError("timeout"))),
        RunnableLambda(lambda _: ""),
        RunnableLambda(lambda _: f"leak {PROMPT_CANARY}"),
    ],
)
async def test_local_answer_falls_back_to_curated_text(personalize: Any) -> None:
    """Test LocalAnswerAgent falls back to curated text when personalization fails.

    Args:
        personalize: Personalization chain that may fail, return empty, or leak prompt.
    """
    answer = await LocalAnswerAgent(personalize, OutputGuard()).respond("q", LOCAL)
    assert answer.answer == ITEM.answer


async def test_local_answer_without_personalisation() -> None:
    """Test LocalAnswerAgent returns curated answer when no personalization is configured."""
    answer = await LocalAnswerAgent(None, OutputGuard()).respond("q", LOCAL)
    assert answer.answer == ITEM.answer


async def test_llm_answer_passes_output_guard() -> None:
    """Test LLMAnswerAgent passes safe output through output guard."""
    agent = LLMAnswerAgent(RunnableLambda(lambda _: "Clear your cache."), OutputGuard())
    answer = await agent.respond("q", LLM)
    assert answer.source is AnswerSource.OPENAI
    assert answer.matched_question is None
    assert answer.answer == "Clear your cache."


async def test_llm_answer_unsafe_output_is_replaced() -> None:
    """Test LLMAnswerAgent replaces unsafe output containing API keys."""
    agent = LLMAnswerAgent(RunnableLambda(lambda _: "key: sk-" + "a" * 40), OutputGuard())
    assert (await agent.respond("q", LLM)).answer == SAFE_FALLBACK_MESSAGE


async def test_llm_answer_provider_errors_are_translated() -> None:
    """Test LLMAnswerAgent translates provider errors to domain exceptions."""

    class RateLimitError(Exception):
        pass

    with pytest.raises(UpstreamRateLimitError):
        await LLMAnswerAgent(RunnableLambda(_raise(RateLimitError())), OutputGuard()).respond(
            "q", LLM
        )
    with pytest.raises(UpstreamServiceError):
        await LLMAnswerAgent(RunnableLambda(_raise(OSError())), OutputGuard()).respond("q", LLM)


async def test_compliance_agent_returns_fixed_message() -> None:
    """Test ComplianceAgent returns fixed compliance message for out-of-scope questions."""
    decision = RouteDecision(route=Route.COMPLIANCE, reason="out_of_scope", decided_by="t")
    answer = await ComplianceAgent().respond("lasagna recipe?", decision)
    assert answer.source is AnswerSource.COMPLIANCE
    assert answer.answer == COMPLIANCE_MESSAGE
