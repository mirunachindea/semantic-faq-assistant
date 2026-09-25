"""Hybrid semantic search over the knowledge base.

Scoring
-------
1. **Dense**: the query embedding is compared (cosine) against *two* representations per FAQ
   entry — the question alone and a ``topic + question + answer`` document — and the max is
   kept. Short, keyword-style questions ("Edit avatar?") gain recall from the answer text,
   while well-phrased questions still match question-to-question.
2. **Lexical boost**: token overlap between the query and the FAQ question/category
   (stop-words removed, light stemming). It can only *raise* a score, never lower it::

       score = dense + w * lexical * (1 - dense)

   so paraphrases with zero word overlap are not penalised, exact-term matches (e.g. "2FA",
   "passkeys", "invoices") are rewarded, and the result stays within ``[dense, 1]`` which
   keeps routing thresholds interpretable.
"""

import logging
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from faq_assistant.domain.models import SearchHit
from faq_assistant.knowledge_base.cleaning import to_search_text, tokenize
from faq_assistant.llm.errors import upstream_call
from faq_assistant.storage.base import VectorStore

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "get",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "so",
        "the",
        "there",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_MIN_STEM = 3


def _stem(token: str) -> str:
    """Very light stemmer: plurals, -ing/-ed and a trailing -e (change/changed/changing)."""
    if token.endswith("ies") and len(token) > _MIN_STEM + 2:
        token = token[:-3] + "y"
    elif token.endswith("s") and not token.endswith("ss") and len(token) > _MIN_STEM + 1:
        token = token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM:
            token = token[: -len(suffix)]
            break
    if token.endswith("e") and len(token) > _MIN_STEM + 1:
        token = token[:-1]
    return token


def content_terms(text: str) -> set[str]:
    """Stemmed, stop-word-free terms of ``text``."""
    return {_stem(t) for t in tokenize(text) if t not in _STOPWORDS}


def lexical_overlap(query: str, document: str) -> float:
    """Fraction of query terms present in ``document`` (0 when the query has no terms)."""
    query_terms = content_terms(query)
    if not query_terms:
        return 0.0
    return len(query_terms & content_terms(document)) / len(query_terms)


def combine_scores(dense: float, lexical: float, weight: float) -> float:
    """Bounded lexical boost on top of the dense score (see module docstring)."""
    dense = max(0.0, min(1.0, dense))
    return dense + weight * lexical * (1.0 - dense)


class HybridSearcher:
    """Embeds a query, retrieves dense candidates and re-ranks them with a lexical boost."""

    def __init__(
        self,
        store: VectorStore,
        embeddings: Embeddings,
        *,
        top_k: int = 5,
        lexical_weight: float = 0.2,
    ):
        self._store = store
        self._embeddings = embeddings
        self._top_k = top_k
        self._lexical_weight = lexical_weight

    async def search(self, query: str, collection: str) -> list[SearchHit]:
        """Return hits ordered by descending combined score."""
        search_text = to_search_text(query)
        async with upstream_call("query embedding"):
            vector: Sequence[float] = await self._embeddings.aembed_query(search_text)
        candidates = await self._store.search(collection, vector, self._top_k)
        hits = []
        for item, dense in candidates:
            lexical = lexical_overlap(search_text, f"{item.question_text} {item.category}")
            hits.append(
                SearchHit(
                    item=item,
                    dense_score=round(dense, 4),
                    lexical_score=round(lexical, 4),
                    score=round(combine_scores(dense, lexical, self._lexical_weight), 4),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        if hits:
            logger.info(
                "Top hit %.3f (dense %.3f) for question %r",
                hits[0].score,
                hits[0].dense_score,
                hits[0].item.question,
            )
        return hits
