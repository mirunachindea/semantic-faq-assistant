from pathlib import Path

from faq_assistant.container import Container
from faq_assistant.evaluation import (
    EvalCase,
    PrefetchedQueryEmbeddings,
    evaluate_retrieval,
    evaluate_routing,
    load_eval_cases,
)
from tests.fakes import BagOfWordsEmbeddings

DATASET = Path(__file__).resolve().parents[2] / "data" / "eval" / "eval_set.jsonl"

CASES = [
    EvalCase(
        query="download my invoices",
        expected_question="Where can I download invoices?",
        expected_route="local",
    ),
    EvalCase(query="best lasagna recipe", expected_route="compliance"),
]


def test_bundled_dataset_parses() -> None:
    """Test evaluation dataset loads and contains expected routing categories."""
    cases = load_eval_cases(DATASET)
    assert len(cases) > 30
    assert {c.expected_route for c in cases} == {"local", "llm", "compliance"}


async def test_retrieval_metrics(container: Container) -> None:
    """Test retrieval evaluation computes ranking metrics.

    Args:
        container: Test container fixture with searcher.
    """
    report = await evaluate_retrieval(container.searcher, "faq", CASES, thresholds=(0.0, 0.99))
    assert report.cases == 1
    assert report.hit_at_1 == 1.0
    assert report.mrr == 1.0
    at_zero = report.threshold_sweep[0]
    assert at_zero == {"threshold": 0.0, "correct": 1, "wrong_match": 0, "not_in_kb": 1}


async def test_routing_metrics(container: Container) -> None:
    """Test routing evaluation computes accuracy and confusion matrix.

    Args:
        container: Test container fixture with assistant.
    """
    report = await evaluate_routing(container.assistant, CASES)
    assert report.cases == 2
    assert report.match_accuracy == 1.0
    assert report.confusion["local->local"] == 1


async def test_prefetch_batches_queries_into_one_call() -> None:
    """Test PrefetchedQueryEmbeddings batches and deduplicates prefetch calls."""
    inner = BagOfWordsEmbeddings()
    calls: list[int] = []
    original = inner.embed_documents

    def counting(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return original(texts)

    inner.embed_documents = counting  # type: ignore[method-assign]
    wrapper = PrefetchedQueryEmbeddings(inner)
    await wrapper.prefetch(["reset password!!!", "cancel plan", "cancel plan"])
    assert calls == [2]
    assert await wrapper.aembed_query("reset password!") == inner.embed_query("reset password!")
