from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Literal, Mapping, Sequence

from tracemem.domain import Candidate
from tracemem.text import lexical_text


QueryKind = Literal[
    "fact",
    "current_state",
    "historical",
    "change",
    "causal",
    "multi_hop",
    "summary",
]


@dataclass(frozen=True)
class QueryPlan:
    kind: QueryKind
    lexical_queries: tuple[str, ...]
    semantic_query: str
    entity_terms: tuple[str, ...]
    requested_time: datetime | None
    channel_weights: Mapping[str, float]
    candidate_limit: int
    final_limit: int


class HeuristicQueryRouter:
    _CURRENT = (" now", "currently", "current ", "现在", "目前", "如今")
    _CHANGE = ("change", "changed", "before and after", "变化", "改变", "先后")
    _CAUSAL = ("why", "because", "cause", "reason", "为什么", "原因", "导致")
    _SUMMARY = ("summarize", "summary", "overall", "总结", "概括")
    _MULTI_HOP = ("relationship", "related", "through", "关系", "之间")

    def plan(
        self,
        query: str,
        options: Sequence[str] | None,
    ) -> QueryPlan:
        lowered = f" {query.casefold()} "
        requested_time = self._requested_time(query)
        if any(marker in lowered for marker in self._CHANGE):
            kind: QueryKind = "change"
        elif any(marker in lowered for marker in self._CURRENT):
            kind = "current_state"
        elif requested_time is not None or any(
            marker in lowered
            for marker in (" before", "previous", "formerly", "过去", "以前", "当时")
        ):
            kind = "historical"
        elif any(marker in lowered for marker in self._CAUSAL):
            kind = "causal"
        elif any(marker in lowered for marker in self._SUMMARY):
            kind = "summary"
        elif any(marker in lowered for marker in self._MULTI_HOP):
            kind = "multi_hop"
        else:
            kind = "fact"

        weights = {
            "episode_bm25": 1.0,
            "episode_vector": 1.0,
            "card_bm25": 1.15,
            "card_vector": 1.15,
            "state": 1.0,
        }
        if kind in {"current_state", "historical", "change"}:
            weights["state"] = 2.2
            weights["card_bm25"] = 1.35
            weights["card_vector"] = 1.35
        elif kind == "causal":
            weights["card_bm25"] = 1.5
            weights["card_vector"] = 1.5
        elif kind == "summary":
            weights["episode_vector"] = 1.4

        lexical_queries = (query, *(options or ()))
        stopwords = {
            "where",
            "does",
            "did",
            "what",
            "when",
            "how",
            "why",
            "live",
            "now",
            "current",
            "currently",
            "in",
            "the",
            "is",
            "was",
        }
        terms = tuple(
            dict.fromkeys(
                token
                for token in lexical_text(query).split()
                if token not in stopwords and len(token) > 1
            )
        )
        return QueryPlan(
            kind=kind,
            lexical_queries=tuple(lexical_queries),
            semantic_query=(
                query
                if not options
                else query + "\nOptions:\n" + "\n".join(options)
            ),
            entity_terms=terms,
            requested_time=requested_time,
            channel_weights=weights,
            candidate_limit=40,
            final_limit=30,
        )

    @staticmethod
    def _requested_time(query: str) -> datetime | None:
        match = re.search(r"\b(19|20)\d{2}\b", query)
        if not match:
            return None
        return datetime(int(match.group(0)), 1, 1, tzinfo=timezone.utc)


class EvidencePacker:
    def __init__(self, *, pairing_enabled: bool = True) -> None:
        self.pairing_enabled = pairing_enabled

    def pack(
        self,
        *,
        plan: QueryPlan,
        candidates: Sequence[Candidate],
        limit: int,
    ) -> list[Candidate]:
        adjusted = [self._adjust(plan, candidate) for candidate in candidates]
        adjusted.sort(
            key=lambda item: (
                -item.score,
                0 if item.is_current else 1,
                item.id,
            )
        )
        unique: list[Candidate] = []
        seen: set[str] = set()
        for candidate in adjusted:
            fingerprint = lexical_text(candidate.content)
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(candidate)

        if self.pairing_enabled and plan.kind == "change":
            paired: list[Candidate] = []
            state_keys: set[str] = set()
            for candidate in unique:
                if candidate.state_key and candidate.state_key not in state_keys:
                    group = [
                        item
                        for item in unique
                        if item.state_key == candidate.state_key
                    ]
                    current = [item for item in group if item.is_current]
                    history = [item for item in group if item.is_current is False]
                    paired.extend(current[:1] + history[:1])
                    state_keys.add(candidate.state_key)
                elif candidate.state_key is None:
                    paired.append(candidate)
            paired_ids = {candidate.id for candidate in paired}
            paired.extend(item for item in unique if item.id not in paired_ids)
            unique = paired
        return unique[:limit]

    @staticmethod
    def _adjust(plan: QueryPlan, candidate: Candidate) -> Candidate:
        score = candidate.score
        temporal_role = candidate.temporal_role
        if candidate.is_current is True:
            temporal_role = "current"
        elif candidate.is_current is False:
            temporal_role = "historical"

        if plan.kind == "current_state":
            if candidate.is_current is True:
                score += 2.0
            elif candidate.is_current is False:
                score -= 0.25
        elif plan.kind == "historical" and plan.requested_time:
            if candidate.event_time:
                if candidate.event_time.year == plan.requested_time.year:
                    score += 2.0
                elif candidate.event_time > plan.requested_time:
                    score -= 0.5
        elif plan.kind == "change":
            if candidate.is_current is True:
                score += 1.5
            elif candidate.is_current is False:
                score += 1.0
        return replace(candidate, score=score, temporal_role=temporal_role)


def render_candidate(candidate: Candidate) -> str:
    kind = candidate.memory_kind or candidate.source_type
    parts = [f"Memory: {kind}"]
    if candidate.event_time:
        parts.append(f"event time: {candidate.event_time.date().isoformat()}")
    parts.append(f"learned: {candidate.created_at.date().isoformat()}")
    if candidate.temporal_role:
        parts.append(f"status: {candidate.temporal_role}")
    header = "[" + " | ".join(parts) + "]"
    return f"{header}\n{candidate.content}"
