"""Offline evaluation of retrieval quality and routing accuracy.

Dataset format (JSONL), one case per line::

    {"query": "...", "expected_question": "<FAQ question>" | null, "expected_route": "local"}

* Retrieval metrics (embeddings only, cheap): hit@1, hit@k, MRR over cases with an
  ``expected_question``, plus a threshold sweep showing how many cases would be answered
  locally *and correctly* vs. wrongly at each score cut-off.
* Routing metrics (``with_router=True``, calls the chat model): end-to-end route accuracy and
  correct-match rate through the full assistant.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from faq_assistant.domain.models import AnswerSource
from faq_assistant.knowledge_base.cleaning import normalize_text, to_search_text
from faq_assistant.retrieval.search import HybridSearcher
from faq_assistant.services.assistant import FAQAssistant

_ROUTE_TO_SOURCE = {
    "local": AnswerSource.LOCAL,
    "llm": AnswerSource.OPENAI,
    "compliance": AnswerSource.COMPLIANCE,
}


class PrefetchedQueryEmbeddings(Embeddings):
    """Embeddings wrapper that serves query vectors from a batch-prefetched cache.

    Evaluating N queries then costs one embedding request instead of N, which matters with
    rate-limited keys (and is equivalent: OpenAI embeds queries and documents identically).
    """

    def __init__(self, inner: Embeddings):
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    async def prefetch(self, queries: Sequence[str]) -> None:
        """Embed ``queries`` (with the searcher's preprocessing) in a single batch."""
        texts = sorted({to_search_text(q) for q in queries} - self._cache.keys())
        if texts:
            vectors = await self._inner.aembed_documents(texts)
            self._cache.update(zip(texts, vectors, strict=True))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Delegate to the wrapped model."""
        return self._inner.embed_documents(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Delegate to the wrapped model."""
        return await self._inner.aembed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Serve from cache, falling back to the wrapped model."""
        return self._cache.get(text) or self._inner.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        """Serve from cache, falling back to the wrapped model."""
        return self._cache.get(text) or await self._inner.aembed_query(text)


class EvalCase(BaseModel):
    """One labelled query."""

    query: str
    expected_question: str | None = None
    expected_route: Literal["local", "llm", "compliance"]


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Read a JSONL evaluation dataset (blank lines and ``#`` comments are ignored)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        EvalCase.model_validate(json.loads(line))
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _same(a: str | None, b: str | None) -> bool:
    return (
        a is not None
        and b is not None
        and normalize_text(a).casefold() == normalize_text(b).casefold()
    )


@dataclass(slots=True)
class RetrievalReport:
    """Retrieval metrics."""

    cases: int = 0
    hit_at_1: float = 0.0
    hit_at_k: float = 0.0
    mrr: float = 0.0
    # Per threshold: how many top-1 hits would be answered locally, split into correct
    # answers, wrong FAQ entries, and questions that have no KB answer at all.
    threshold_sweep: list[dict[str, float]] = field(default_factory=list)
    misses: list[dict[str, object]] = field(default_factory=list)


async def evaluate_retrieval(
    searcher: HybridSearcher,
    collection: str,
    cases: Sequence[EvalCase],
    thresholds: Sequence[float] = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85),
) -> RetrievalReport:
    """Compute ranking metrics and a local-answer threshold sweep."""
    report = RetrievalReport()
    # (top-1 score, case has a KB answer, top-1 is that answer)
    top1: list[tuple[float, bool, bool]] = []
    reciprocal_ranks: list[float] = []
    hits1 = hitsk = 0
    for case in cases:
        hits = await searcher.search(case.query, collection)
        best_correct = bool(hits) and _same(hits[0].item.question, case.expected_question)
        top1.append(
            (hits[0].score if hits else 0.0, case.expected_question is not None, best_correct)
        )
        if case.expected_question is None:
            continue
        report.cases += 1
        rank = next(
            (i for i, h in enumerate(hits, 1) if _same(h.item.question, case.expected_question)),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        hits1 += rank == 1
        hitsk += rank is not None
        if rank != 1:
            report.misses.append(
                {
                    "query": case.query,
                    "expected": case.expected_question,
                    "got": hits[0].item.question if hits else None,
                    "score": hits[0].score if hits else None,
                    "rank": rank,
                }
            )
    if report.cases:
        report.hit_at_1 = hits1 / report.cases
        report.hit_at_k = hitsk / report.cases
        report.mrr = sum(reciprocal_ranks) / report.cases
    for threshold in thresholds:
        accepted = [(positive, ok) for score, positive, ok in top1 if score >= threshold]
        report.threshold_sweep.append(
            {
                "threshold": threshold,
                "correct": sum(ok for _, ok in accepted),
                "wrong_match": sum(positive and not ok for positive, ok in accepted),
                "not_in_kb": sum(not positive for positive, _ in accepted),
            }
        )
    return report


@dataclass(slots=True)
class RoutingReport:
    """End-to-end routing metrics."""

    cases: int = 0
    route_accuracy: float = 0.0
    match_accuracy: float = 0.0
    confusion: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, object]] = field(default_factory=list)


async def evaluate_routing(assistant: FAQAssistant, cases: Sequence[EvalCase]) -> RoutingReport:
    """Run every case through the full assistant (uses the chat model)."""
    report = RoutingReport(cases=len(cases))
    confusion: Counter[str] = Counter()
    correct_route = correct_match = local_cases = 0
    for case in cases:
        answer = await assistant.ask(case.query)
        expected = _ROUTE_TO_SOURCE[case.expected_route]
        confusion[f"{expected.value}->{answer.source.value}"] += 1
        route_ok = answer.source == expected
        correct_route += route_ok
        if case.expected_route == "local":
            local_cases += 1
            correct_match += _same(answer.matched_question, case.expected_question)
        if not route_ok:
            report.errors.append(
                {
                    "query": case.query,
                    "expected": expected.value,
                    "got": answer.source.value,
                    "reason": answer.route_reason,
                }
            )
    report.route_accuracy = correct_route / len(cases) if cases else 0.0
    report.match_accuracy = correct_match / local_cases if local_cases else 0.0
    report.confusion = dict(confusion)
    return report
