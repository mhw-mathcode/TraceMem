from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from tracemem.api_models import SearchRequest, SearchResponse, SearchResult
from tracemem.evidence import (
    EvidencePacker,
    HeuristicQueryRouter,
    render_candidate,
)
from tracemem.model_clients import DisabledReranker, ModelError, Reranker
from tracemem.observability import duration_ms, log_event
from tracemem.retrieval import HybridRetriever


logger = logging.getLogger(__name__)


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
        search_id = uuid4().hex[:12]
        started = perf_counter()
        stage = "planning"
        log_event(
            logger,
            logging.INFO,
            "search_started",
            search_id=search_id,
            top_k=request.top_k,
            options=len(request.options or ()),
        )
        try:
            plan = self.router.plan(request.query, request.options)

            stage = "retrieval"
            retrieval_started = perf_counter()
            candidates = await self.retriever.retrieve(
                user_id=request.user_id,
                query=request.query,
                options=request.options,
                plan=plan,
                search_id=search_id,
            )
            log_event(
                logger,
                logging.INFO,
                "search_retrieved",
                search_id=search_id,
                candidates=len(candidates),
                duration_ms=duration_ms(retrieval_started),
            )

            stage = "rerank"
            rerank_started = perf_counter()
            try:
                rerank_candidates = candidates[:40]
                reranked = await self.reranker.rerank(
                    request.query,
                    rerank_candidates,
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
                log_event(
                    logger,
                    logging.INFO,
                    "search_reranked",
                    search_id=search_id,
                    candidates=len(rerank_candidates),
                    duration_ms=duration_ms(rerank_started),
                )
            except ModelError as error:
                log_event(
                    logger,
                    logging.WARNING,
                    "search_degraded",
                    search_id=search_id,
                    stage="rerank",
                    error=type(error).__name__,
                    duration_ms=duration_ms(rerank_started),
                )

            stage = "packing"
            limit = min(request.top_k, self.final_result_limit, plan.final_limit)
            selected = self.packer.pack(
                plan=plan,
                candidates=candidates,
                limit=limit,
            )
            if not selected:
                log_event(
                    logger,
                    logging.INFO,
                    "search_completed",
                    search_id=search_id,
                    results=0,
                    duration_ms=duration_ms(started),
                )
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
            log_event(
                logger,
                logging.INFO,
                "search_completed",
                search_id=search_id,
                results=len(results),
                duration_ms=duration_ms(started),
            )
            return SearchResponse(data=results)
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "search_failed",
                search_id=search_id,
                stage=stage,
                error=type(error).__name__,
                duration_ms=duration_ms(started),
            )
            raise
