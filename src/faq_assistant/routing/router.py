"""Semantic router: an ordered chain of routing rules.

Each rule inspects a :class:`RoutingContext` and either returns a :class:`RouteDecision` or
abstains (``None``), passing control to the next rule. Adding a new policy (e.g. "billing
questions from free-tier users go to a human", "questions in German go to another
collection") means writing one small class and inserting it in the chain; nothing else
changes.

Default chain (see :func:`default_rules`)::

    InputGuardRule           -> compliance   (cheap, deterministic; no retrieval/model cost)
    HighConfidenceMatchRule  -> local        (clear KB hit: skip the LLM, save tokens/latency)
    LLMRouterRule            -> local | llm | compliance   (agentic decision over candidates)
    ScoreFallbackRule        -> local | llm  (deterministic backup if the LLM router abstains)
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from langchain_core.runnables import Runnable

from faq_assistant.domain.models import Route, RouteDecision, SearchHit
from faq_assistant.guardrails.input_guard import InputGuard
from faq_assistant.llm.chains import RouterVerdict
from faq_assistant.llm.errors import upstream_call

logger = logging.getLogger(__name__)

Retriever = Callable[[str], Awaitable[list[SearchHit]]]


class RoutingContext:
    """Per-request routing state. Retrieval runs lazily, at most once."""

    def __init__(self, question: str, retriever: Retriever):
        """Initialize the routing context.

        Args:
            question: The user's question to route.
            retriever: Async function to retrieve candidates for the question.
        """
        self.question = question
        self._retriever = retriever
        self._hits: list[SearchHit] | None = None

    async def hits(self) -> list[SearchHit]:
        """Retrieved candidates, best first.

        Returns:
            A list of SearchHit objects sorted by score, retrieved once and cached.
        """
        if self._hits is None:
            self._hits = await self._retriever(self.question)
        return self._hits

    async def best_hit(self) -> SearchHit | None:
        """Top candidate, if any.

        Returns:
            The top SearchHit or None if no hits are available.
        """
        hits = await self.hits()
        return hits[0] if hits else None


class RoutingRule(Protocol):
    """A single routing policy."""

    name: str

    async def evaluate(self, context: RoutingContext) -> RouteDecision | None:
        """Return a decision, or ``None`` to defer to the next rule."""
        ...


class InputGuardRule:
    """Send inputs flagged by the deterministic input guard to the compliance agent."""

    name = "input_guard"

    def __init__(self, guard: InputGuard):
        """Initialize the input guard rule.

        Args:
            guard: InputGuard instance for checking untrusted input.
        """
        self._guard = guard

    async def evaluate(self, context: RoutingContext) -> RouteDecision | None:
        """Block unsafe input before any retrieval or model call.

        Args:
            context: The routing context containing the question.

        Returns:
            A RouteDecision sending to compliance if input is blocked, None otherwise.
        """
        verdict = self._guard.check(context.question)
        if verdict.allowed:
            return None
        logger.warning("Input blocked by guardrail: %s", verdict.reason)
        return RouteDecision(
            route=Route.COMPLIANCE, reason=verdict.reason or "blocked", decided_by=self.name
        )


class HighConfidenceMatchRule:
    """Answer locally when the best match is unambiguous."""

    name = "high_confidence_match"

    def __init__(self, threshold: float):
        """Initialize the high confidence match rule.

        Args:
            threshold: Minimum similarity score to accept a match without LLM router.
        """
        self._threshold = threshold

    async def evaluate(self, context: RoutingContext) -> RouteDecision | None:
        """Accept the top hit if its score clears the threshold.

        Args:
            context: The routing context containing retrieval results.

        Returns:
            A RouteDecision sending to local if threshold is met, None otherwise.
        """
        best = await context.best_hit()
        if best is None or best.score < self._threshold:
            return None
        return RouteDecision(
            route=Route.LOCAL,
            hit=best,
            reason=f"score {best.score:.3f} >= {self._threshold}",
            decided_by=self.name,
        )


class LLMRouterRule:
    """Let an LLM judge intent, scope and safety given the retrieved candidates.

    The model output is treated as untrusted: it must parse into :class:`RouterVerdict`
    and any referenced candidate must be one we actually offered. Any failure (provider
    error, malformed output, hallucinated id) makes the rule abstain instead of guessing.
    """

    name = "llm_router"

    def __init__(
        self,
        chain: Runnable[dict[str, str], RouterVerdict],
        *,
        candidate_threshold: float,
        max_candidates: int = 3,
    ):
        """Initialize the LLM router rule.

        Args:
            chain: LLM chain to invoke for routing decisions.
            candidate_threshold: Minimum score to include a hit as a candidate.
            max_candidates: Maximum number of candidates to present to the LLM.
        """
        self._chain = chain
        self._candidate_threshold = candidate_threshold
        self._max_candidates = max_candidates

    @staticmethod
    def _format_candidates(candidates: Sequence[tuple[str, SearchHit]]) -> str:
        """Format candidates for presentation to the LLM router.

        Args:
            candidates: List of (candidate_id, SearchHit) tuples.

        Returns:
            A formatted string representation of the candidates for the LLM.
        """
        if not candidates:
            return "(none)"
        return "\n\n".join(
            f"id: {key}\ncategory: {hit.item.category}\nquestion: {hit.item.question}\n"
            f"answer: {hit.item.answer[:300]}"
            for key, hit in candidates
        )

    async def evaluate(self, context: RoutingContext) -> RouteDecision | None:
        """Ask the LLM router for a decision and validate it.

        Args:
            context: The routing context containing the question and retrieval results.

        Returns:
            A RouteDecision based on the LLM's verdict, or None if validation fails.
        """
        hits = await context.hits()
        # Short synthetic ids ("c1") instead of UUIDs: fewer tokens, fewer copy errors.
        candidates = [
            (f"c{i}", hit)
            for i, hit in enumerate(
                [h for h in hits if h.score >= self._candidate_threshold][: self._max_candidates],
                start=1,
            )
        ]
        try:
            async with upstream_call("LLM router"):
                verdict = await self._chain.ainvoke(
                    {
                        "question": context.question,
                        "candidates": self._format_candidates(candidates),
                    }
                )
        except Exception:
            logger.warning("LLM router unavailable; deferring to fallback rules")
            return None
        if not isinstance(verdict, RouterVerdict):
            logger.warning("LLM router returned malformed output: %r", type(verdict))
            return None

        logger.info("LLM router verdict: %s", verdict.model_dump())
        if verdict.is_prompt_injection or verdict.route == "compliance":
            reason = "prompt_injection" if verdict.is_prompt_injection else "out_of_scope"
            return RouteDecision(
                route=Route.COMPLIANCE,
                reason=f"{reason}: {verdict.reasoning}",
                decided_by=self.name,
            )
        if verdict.route == "llm":
            return RouteDecision(route=Route.LLM, reason=verdict.reasoning, decided_by=self.name)

        chosen = dict(candidates).get(verdict.faq_id or "")
        if chosen is None:
            logger.warning("LLM router chose unknown candidate %r", verdict.faq_id)
            return None
        return RouteDecision(
            route=Route.LOCAL, hit=chosen, reason=verdict.reasoning, decided_by=self.name
        )


class ScoreFallbackRule:
    """Deterministic threshold decision, used when the LLM router abstains."""

    name = "score_fallback"

    def __init__(self, threshold: float):
        """Initialize the score fallback rule.

        Args:
            threshold: Minimum score to route to local instead of LLM.
        """
        self._threshold = threshold

    async def evaluate(self, context: RoutingContext) -> RouteDecision | None:
        """Local if the best score clears the threshold, otherwise the LLM.

        Args:
            context: The routing context containing retrieval results.

        Returns:
            A RouteDecision routing to local or LLM based on score threshold.
        """
        best = await context.best_hit()
        if best is not None and best.score >= self._threshold:
            return RouteDecision(
                route=Route.LOCAL,
                hit=best,
                reason=f"fallback: score {best.score:.3f} >= {self._threshold}",
                decided_by=self.name,
            )
        return RouteDecision(
            route=Route.LLM, reason="fallback: no sufficiently similar FAQ", decided_by=self.name
        )


class SemanticRouter:
    """Runs routing rules in order; the first non-``None`` decision wins."""

    def __init__(self, rules: Sequence[RoutingRule]):
        """Initialize the semantic router.

        Args:
            rules: Sequence of routing rules to evaluate in order.
        """
        if not rules:
            msg = "SemanticRouter needs at least one rule"
            raise ValueError(msg)
        self._rules = tuple(rules)

    @property
    def rule_names(self) -> list[str]:
        """Names of the configured rules, in evaluation order."""
        return [rule.name for rule in self._rules]

    async def route(self, context: RoutingContext) -> RouteDecision:
        """Return the decision of the first rule that does not abstain.

        Args:
            context: The routing context containing question and retrieval results.

        Returns:
            A RouteDecision from the first matching rule or default LLM route.
        """
        for rule in self._rules:
            decision = await rule.evaluate(context)
            if decision is not None:
                logger.info(
                    "Routed to %s by %s (%s)", decision.route, decision.decided_by, decision.reason
                )
                return decision
        return RouteDecision(route=Route.LLM, reason="no rule matched", decided_by="default")


def default_rules(
    *,
    input_guard: InputGuard,
    router_chain: Runnable[dict[str, str], RouterVerdict],
    accept_threshold: float,
    candidate_threshold: float,
    fallback_accept_threshold: float,
) -> list[RoutingRule]:
    """The production rule chain.

    Args:
        input_guard: Guard for blocking unsafe input.
        router_chain: LLM chain for semantic routing decisions.
        accept_threshold: Score for automatic local routing (high confidence).
        candidate_threshold: Minimum score for presenting a hit to the LLM.
        fallback_accept_threshold: Score fallback for routing when LLM router abstains.

    Returns:
        A list of RoutingRule instances in the standard evaluation order.
    """
    return [
        InputGuardRule(input_guard),
        HighConfidenceMatchRule(accept_threshold),
        LLMRouterRule(router_chain, candidate_threshold=candidate_threshold),
        ScoreFallbackRule(fallback_accept_threshold),
    ]
