from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from tracemem.db import Database
from tracemem.domain import Candidate, VectorRecord
from tracemem.evidence import QueryPlan
from tracemem.model_clients import Embedder
from tracemem.text import lexical_text


def cosine_candidates(
    query_vector: np.ndarray,
    records: Sequence[VectorRecord],
    *,
    limit: int,
) -> list[Candidate]:
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0 or limit <= 0:
        return []
    compatible = [record for record in records if record.vector.size == query.size]
    if not compatible:
        return []
    matrix = np.stack(
        [np.asarray(record.vector, dtype=np.float32) for record in compatible]
    )
    row_norms = np.linalg.norm(matrix, axis=1)
    valid = row_norms > 0
    similarities = np.full(len(compatible), -1.0, dtype=np.float32)
    similarities[valid] = (
        matrix[valid] @ query / (row_norms[valid] * query_norm)
    )
    ordered_indexes = np.argsort(-similarities, kind="stable")[:limit]
    return [
        replace(
            compatible[int(index)].candidate,
            score=float(similarities[int(index)]),
        )
        for index in ordered_indexes
        if similarities[int(index)] > -1.0
    ]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Candidate]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[Candidate]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists")
    totals: dict[str, float] = {}
    best: dict[str, tuple[int, Candidate]] = {}
    for channel, weight in zip(ranked_lists, weights, strict=True):
        for rank, item in enumerate(channel, start=1):
            totals[item.id] = totals.get(item.id, 0.0) + weight / (k + rank)
            previous = best.get(item.id)
            if previous is None or rank < previous[0]:
                best[item.id] = (rank, item)
    ordered_ids = sorted(
        totals,
        key=lambda item_id: (-totals[item_id], best[item_id][0], item_id),
    )
    return [
        replace(best[item_id][1], score=totals[item_id])
        for item_id in ordered_ids
    ]


def fts_query(value: str) -> str | None:
    tokens = lexical_text(value).split()
    if not tokens:
        return None
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


class HybridRetriever:
    def __init__(
        self,
        *,
        database: Database,
        embedder: Embedder,
        cards_enabled: bool = False,
        state_enabled: bool = False,
        per_channel_limit: int = 40,
    ) -> None:
        self.database = database
        self.embedder = embedder
        self.cards_enabled = cards_enabled
        self.state_enabled = state_enabled
        self.per_channel_limit = per_channel_limit

    async def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        options: Sequence[str] | None = None,
        plan: QueryPlan | None = None,
    ) -> list[Candidate]:
        semantic_query = query
        if options:
            semantic_query += "\nOptions:\n" + "\n".join(options)
        lexical_query = fts_query(semantic_query)
        if not lexical_query and not lexical_text(semantic_query):
            return []

        query_vectors = await self.embedder.embed([semantic_query])
        if len(query_vectors) != 1:
            raise RuntimeError("query embedder returned the wrong vector count")

        channels: list[list[Candidate]] = []
        weights: list[float] = []
        channel_weights = (
            plan.channel_weights
            if plan
            else {
                "episode_bm25": 1.0,
                "episode_vector": 1.0,
                "card_bm25": 1.15,
                "card_vector": 1.15,
                "state": 1.0,
            }
        )
        if lexical_query:
            channels.append(
                self.database.search_episode_fts(
                    user_id,
                    lexical_query,
                    self.per_channel_limit,
                )
            )
            weights.append(channel_weights["episode_bm25"])
        channels.append(
            cosine_candidates(
                query_vectors[0],
                self.database.load_episode_vectors(user_id),
                limit=self.per_channel_limit,
            )
        )
        weights.append(channel_weights["episode_vector"])

        if self.cards_enabled:
            if lexical_query:
                channels.append(
                    self.database.search_card_fts(
                        user_id,
                        lexical_query,
                        self.per_channel_limit,
                    )
                )
                weights.append(channel_weights["card_bm25"])
            channels.append(
                cosine_candidates(
                    query_vectors[0],
                    self.database.load_card_vectors(user_id),
                    limit=self.per_channel_limit,
                )
            )
            weights.append(channel_weights["card_vector"])

        if self.state_enabled:
            states = self.database.load_state_candidates(user_id)
            if plan and plan.entity_terms:
                query_terms = set(plan.entity_terms)
                matching = [
                    candidate
                    for candidate in states
                    if query_terms
                    & set(lexical_text(candidate.content).split())
                ]
                if matching:
                    states = matching
            channels.append(states[: self.per_channel_limit])
            weights.append(channel_weights["state"])

        return reciprocal_rank_fusion(channels, weights=weights)
