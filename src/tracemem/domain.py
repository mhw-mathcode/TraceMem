from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

import numpy as np

from tracemem.multimodal import ContentPart


class ClaimStatus(str, Enum):
    CLAIMED = "claimed"
    WAIT = "wait"
    COMPLETED = "completed"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ClaimResult:
    status: ClaimStatus
    request_id: str
    user_id: str
    session_id: str
    lease_until: datetime | None = None


@dataclass(frozen=True)
class EpisodeDraft:
    id: str
    user_id: str
    session_id: str
    request_id: str
    message_index: int
    role: str
    content: str
    lexical_text: str
    event_time: datetime | None
    ingested_at: datetime
    importance: float
    embedding: np.ndarray
    original_content: list[ContentPart] | None = None


@dataclass(frozen=True)
class MemoryCardDraft:
    id: str
    user_id: str
    kind: Literal[
        "fact",
        "event",
        "preference",
        "goal",
        "plan",
        "relationship",
        "causal",
    ]
    subject: str
    predicate: str
    object: str
    event_time: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    ingested_at: datetime
    polarity: Literal["positive", "negative", "uncertain"]
    confidence: float
    importance: float
    status: str
    source_episode_ids: tuple[str, ...]
    search_text: str
    lexical_text: str
    embedding: np.ndarray
    relation_hint: Literal["continue", "update", "correction", "conflict"] | None = None


@dataclass(frozen=True)
class ExtractedCard:
    kind: Literal[
        "fact",
        "event",
        "preference",
        "goal",
        "plan",
        "relationship",
        "causal",
    ]
    subject: str
    predicate: str
    object: str
    event_time: datetime | None
    polarity: Literal["positive", "negative", "uncertain"]
    confidence: float
    importance: float
    source_message_indexes: tuple[int, ...]
    relation_hint: Literal["continue", "update", "correction", "conflict"] | None


@dataclass(frozen=True)
class StateVersionDraft:
    id: str
    user_id: str
    state_key: str
    memory_card_id: str
    relation_to_previous: Literal["continue", "update", "correction", "conflict"]
    previous_version_id: str | None
    valid_from: datetime | None
    valid_to: datetime | None
    is_current: bool


@dataclass(frozen=True)
class Candidate:
    id: str
    source_type: Literal["episode", "card"]
    content: str
    created_at: datetime
    score: float
    event_time: datetime | None = None
    state_key: str | None = None
    is_current: bool | None = None
    memory_kind: str | None = None
    source_episode_ids: tuple[str, ...] = ()
    temporal_role: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorRecord:
    candidate: Candidate
    vector: np.ndarray


@dataclass(frozen=True)
class ExistingStateVersion:
    id: str
    user_id: str
    state_key: str
    memory_card_id: str
    object: str
    event_time: datetime | None
    valid_from: datetime | None
    valid_to: datetime | None
    is_current: bool
