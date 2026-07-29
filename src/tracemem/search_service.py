from __future__ import annotations

from dataclasses import replace

from tracemem.api_models import SearchRequest, SearchResponse, SearchResult
from tracemem.evidence import (
    EvidencePacker,
    HeuristicQueryRouter,
    render_candidate,
)
from tracemem.model_clients import DisabledReranker, ModelError, Reranker
from tracemem.retrieval import HybridRetriever


class SearchService:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        final_result_limit: int = 30,
        router: HeuristicQueryRouter | None = None,
        reranker: Reranker | None = None,
        packer: EvidencePacker | None = None,
    ) -> None:
        self.retriever = retriever
        self.final_result_limit = final_result_limit
        self.router = router or HeuristicQueryRouter()
        self.reranker = reranker or DisabledReranker()
        self.packer = packer or EvidencePacker(pairing_enabled=False)

    async def search(self, request: SearchRequest) -> SearchResponse:
        plan = self.router.plan(request.query, request.options)
        candidates = await self.retriever.retrieve(
            user_id=request.user_id,
            query=request.query,
            options=request.options,
            plan=plan,
        )
        try:
            reranked = await self.reranker.rerank(
                request.query,
                candidates[:40],
            )
            candidates = [
                replace(
                    candidate,
                    score=(
                        0.65 * reranked.get(candidate.id, candidate.score)
                        + 0.35 * candidate.score
                    ),
                )
                for candidate in candidates
            ]
        except ModelError:
            pass

        limit = min(request.top_k, self.final_result_limit, plan.final_limit)
        selected = self.packer.pack(
            plan=plan,
            candidates=candidates,
            limit=limit,
        )
        if not selected:
            return SearchResponse(data=[])
        maximum = max(candidate.score for candidate in selected)
        minimum = min(candidate.score for candidate in selected)
        span = maximum - minimum
        results = [
            SearchResult(
                id=candidate.id,
                content=render_candidate(candidate),
                score=(
                    (candidate.score - minimum) / span
                    if span > 0
                    else 1.0
                ),
                created_at=candidate.created_at,
            )
            for candidate in selected
        ]
        return SearchResponse(data=results)
