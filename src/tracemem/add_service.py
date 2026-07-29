from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone

from tracemem.api_models import AddRequest, AddResponse
from tracemem.db import Database
from tracemem.domain import ClaimStatus, EpisodeDraft, MemoryCardDraft
from tracemem.ledger import Ledger, state_key
from tracemem.model_clients import (
    Embedder,
    Extractor,
    MessageForExtraction,
    ModelError,
)
from tracemem.text import (
    lexical_text,
    payload_hash,
    stable_id,
    timestamp_to_datetime,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AddError(RuntimeError):
    pass


class AddConflict(AddError):
    pass


class AddBusy(AddError):
    pass


class AddDependencyError(AddError):
    pass


class AddService:
    def __init__(
        self,
        *,
        database: Database,
        embedder: Embedder,
        extractor: Extractor,
        clock: Callable[[], datetime] = _utc_now,
        lease_seconds: int = 120,
        wait_seconds: float = 5.0,
        ledger: Ledger | None = None,
    ) -> None:
        self.database = database
        self.embedder = embedder
        self.extractor = extractor
        self.clock = clock
        self.lease_seconds = lease_seconds
        self.wait_seconds = wait_seconds
        self.ledger = ledger or Ledger()

    async def add(self, request: AddRequest) -> AddResponse:
        digest = payload_hash(request)
        deadline = asyncio.get_running_loop().time() + self.wait_seconds
        while True:
            claim = self.database.claim_add(
                request.request_id,
                request.user_id,
                request.session_id,
                digest,
                self.lease_seconds,
            )
            if claim.status is ClaimStatus.COMPLETED:
                return self._response(request)
            if claim.status is ClaimStatus.CONFLICT:
                raise AddConflict("request_id was already used with another payload")
            if claim.status is ClaimStatus.CLAIMED:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AddBusy("an identical Add request is still processing")
            await asyncio.sleep(0.05)

        ingested_at = self.clock().astimezone(timezone.utc)
        try:
            vectors = await self.embedder.embed(
                [message.content for message in request.messages]
            )
            if len(vectors) != len(request.messages):
                raise RuntimeError("embedding count mismatch")
        except Exception as error:
            self.database.mark_add_failed(request.request_id, "embedding_failed")
            raise AddDependencyError("episode embedding failed") from error

        episodes = [
            EpisodeDraft(
                id=stable_id("episode", request.request_id, str(index)),
                user_id=request.user_id,
                session_id=request.session_id,
                request_id=request.request_id,
                message_index=index,
                role=message.role,
                content=message.content,
                lexical_text=lexical_text(message.content),
                event_time=timestamp_to_datetime(message.timestamp),
                ingested_at=ingested_at,
                importance=1.0,
                embedding=vectors[index],
            )
            for index, message in enumerate(request.messages)
        ]

        cards: list[MemoryCardDraft] = []
        state_versions = ()
        close_version_ids = ()
        raw_only = True
        try:
            extracted = await self.extractor.extract(
                [
                    MessageForExtraction(
                        index=index,
                        role=message.role,
                        content=message.content,
                        timestamp=message.timestamp,
                    )
                    for index, message in enumerate(request.messages)
                ]
            )
            if extracted:
                card_texts = [
                    f"{card.subject} {card.predicate} {card.object}"
                    for card in extracted
                ]
                card_vectors = await self.embedder.embed(card_texts)
                if len(card_vectors) != len(extracted):
                    raise RuntimeError("card embedding count mismatch")
                episode_ids = {
                    episode.message_index: episode.id for episode in episodes
                }
                cards = [
                    MemoryCardDraft(
                        id=stable_id(
                            "card",
                            request.request_id,
                            str(card_index),
                            card.subject,
                            card.predicate,
                            card.object,
                        ),
                        user_id=request.user_id,
                        kind=card.kind,
                        subject=card.subject,
                        predicate=card.predicate,
                        object=card.object,
                        event_time=card.event_time,
                        valid_from=card.event_time,
                        valid_to=None,
                        ingested_at=ingested_at,
                        polarity=card.polarity,
                        confidence=card.confidence,
                        importance=card.importance,
                        status="active",
                        source_episode_ids=tuple(
                            episode_ids[index]
                            for index in card.source_message_indexes
                        ),
                        search_text=card_texts[card_index],
                        lexical_text=lexical_text(card_texts[card_index]),
                        embedding=card_vectors[card_index],
                        relation_hint=card.relation_hint,
                    )
                    for card_index, card in enumerate(extracted)
                ]
                keys = [state_key(card.subject, card.predicate) for card in cards]
                plan = self.ledger.plan(
                    existing_versions=self.database.load_current_versions(
                        request.user_id,
                        keys,
                    ),
                    cards=cards,
                )
                state_versions = plan.new_versions
                close_version_ids = plan.close_version_ids
                raw_only = False
        except ModelError:
            raw_only = True
        except (KeyError, RuntimeError):
            cards = []
            state_versions = ()
            close_version_ids = ()
            raw_only = True

        self.database.commit_add(
            request_id=request.request_id,
            episodes=episodes,
            cards=cards,
            state_versions=state_versions,
            close_version_ids=close_version_ids,
            raw_only=raw_only,
        )
        return self._response(request)

    @staticmethod
    def _response(request: AddRequest) -> AddResponse:
        return AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
