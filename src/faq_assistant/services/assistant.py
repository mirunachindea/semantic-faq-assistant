"""Application service orchestrating retrieval, routing and answering."""

import logging
from collections.abc import Mapping

from faq_assistant.agents.responders import Responder
from faq_assistant.domain.models import AssistantAnswer, Route, SearchHit
from faq_assistant.retrieval.search import HybridSearcher
from faq_assistant.routing.router import RoutingContext, SemanticRouter

logger = logging.getLogger(__name__)


class FAQAssistant:
    """Entry point used by the API: question in, answer out."""

    def __init__(
        self,
        searcher: HybridSearcher,
        router: SemanticRouter,
        responders: Mapping[Route, Responder],
        *,
        collection: str,
    ):
        missing = set(Route) - set(responders)
        if missing:
            msg = f"No responder configured for routes: {sorted(missing)}"
            raise ValueError(msg)
        self._searcher = searcher
        self._router = router
        self._responders = dict(responders)
        self._collection = collection

    async def _retrieve(self, question: str) -> list[SearchHit]:
        return await self._searcher.search(question, self._collection)

    async def ask(self, question: str) -> AssistantAnswer:
        """Route ``question`` and produce an answer."""
        context = RoutingContext(question, self._retrieve)
        decision = await self._router.route(context)
        return await self._responders[decision.route].respond(question, decision)
