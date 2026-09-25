"""Deterministic test doubles: no network, no API keys."""

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from faq_assistant.llm.chains import LLMChains, RouterVerdict
from faq_assistant.retrieval.search import content_terms

DIMENSIONS = 256


class BagOfWordsEmbeddings(Embeddings):
    """Hashing bag-of-words embeddings: texts sharing terms get high cosine similarity."""

    def __init__(self, dimensions: int = DIMENSIONS):
        self.dimensions = dimensions
        self.embedded_texts: list[str] = []
        self.fail_with: Exception | None = None

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions)
        for term in content_terms(text):
            digest = hashlib.md5(term.encode(), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        norm = np.linalg.norm(vector)
        return [float(x) for x in (vector / norm if norm else vector)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_with:
            raise self.fail_with
        self.embedded_texts.extend(texts)
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.fail_with:
            raise self.fail_with
        return self._embed(text)


class RecordingRunnable:
    """Wraps a function in a Runnable and records its inputs."""

    def __init__(self, fn: Callable[[dict[str, str]], Any]):
        self.calls: list[dict[str, str]] = []

        def _record(payload: dict[str, str]) -> Any:
            self.calls.append(payload)
            return fn(payload)

        self.runnable = RunnableLambda(_record)


def make_chains(
    router: Callable[[dict[str, str]], Any] | None = None,
    answer: Callable[[dict[str, str]], Any] | None = None,
    personalize: Callable[[dict[str, str]], Any] | None = None,
) -> tuple[LLMChains, dict[str, RecordingRunnable]]:
    """Build fake chains; defaults: router says "llm", answer/personalize echo."""
    recorders = {
        "router": RecordingRunnable(
            router
            or (
                lambda _: RouterVerdict(
                    route="llm", faq_id=None, is_prompt_injection=False, reasoning="not in FAQ"
                )
            )
        ),
        "answer": RecordingRunnable(answer or (lambda p: f"General answer to: {p['question']}")),
        "personalize": RecordingRunnable(
            personalize or (lambda p: f"Personalised: {p['reference_answer']}")
        ),
    }
    chains = LLMChains(
        router=recorders["router"].runnable,
        answer=recorders["answer"].runnable,
        personalize=recorders["personalize"].runnable,
    )
    return chains, recorders
