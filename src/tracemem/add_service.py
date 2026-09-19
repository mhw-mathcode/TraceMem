from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from time import perf_counter

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
from tracemem.multimodal import ImagePart, context_text, index_text
from tracemem.observability import duration_ms, log_event
from tracemem.text import (
    lexical_text,
    payload_hash,
    stable_id,
    timestamp_to_datetime,
)
from tracemem.vision import VisionDescriber, VisionError, VisionUnavailable


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


logger = logging.getLogger(__name__)


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
        vision: VisionDescriber | None = None,
        clock: Callable[[], datetime] = _utc_now,
        lease_seconds: int = 120,
        wait_seconds: float = 5.0,
        ledger: Ledger | None = None,
    ) -> None:
        self.database = database
        self.embedder = embedder
        self.extractor = extractor
        self.vision = vision
        self.clock = clock
        self.lease_seconds = lease_seconds
        self.wait_seconds = wait_seconds
        self.ledger = ledger or Ledger()

    async def add(self, request: AddRequest) -> AddResponse:
        started = perf_counter()
        log_event(
            logger,
            logging.INFO,
            "add_started",
            request_id=request.request_id,
            messages=len(request.messages),
        )
        digest = payload_hash(request)
        deadline = asyncio.get_running_loop().time() + self.wait_seconds
        claim_started = perf_counter()
        claim_attempts = 0
        while True:
            claim_attempts += 1
            try:
                claim = self.database.claim_add(
                    request.request_id,
                    request.user_id,
                    request.session_id,
                    digest,
                    self.lease_seconds,
                )
            except Exception as error:
                log_event(
                    logger,
                    logging.ERROR,
                    "add_failed",
                    request_id=request.request_id,
                    stage="claim",
                    error=type(error).__name__,
                    duration_ms=duration_ms(started),
                )
                raise
            if claim.status is ClaimStatus.COMPLETED:
                log_event(
                    logger,
                    logging.INFO,
                    "add_claimed",
                    request_id=request.request_id,
                    status=claim.status,
                    attempts=claim_attempts,
                    duration_ms=duration_ms(claim_started),
                )
                log_event(
                    logger,
                    logging.INFO,
                    "add_completed",
                    request_id=request.request_id,
                    replayed=True,
                    duration_ms=duration_ms(started),
                )
                return self._response(request)
            if claim.status is ClaimStatus.CONFLICT:
                log_event(
                    logger,
                    logging.WARNING,
                    "add_failed",
                    request_id=request.request_id,
                    stage="claim",
                    error="AddConflict",
                    duration_ms=duration_ms(started),
                )
                raise AddConflict("request_id was already used with another payload")
            if claim.status is ClaimStatus.CLAIMED:
                break
            if asyncio.get_running_loop().time() >= deadline:
                log_event(
                    logger,
                    logging.WARNING,
                    "add_failed",
                    request_id=request.request_id,
                    stage="claim",
                    error="AddBusy",
                    duration_ms=duration_ms(started),
                )
                raise AddBusy("an identical Add request is still processing")
            await asyncio.sleep(0.05)

        log_event(
            logger,
            logging.INFO,
            "add_claimed",
            request_id=request.request_id,
            status=claim.status,
            attempts=claim_attempts,
            duration_ms=duration_ms(claim_started),
        )

        ingested_at = self.clock().astimezone(timezone.utc)
        indexed_texts: list[str] = []
        try:
            for message in request.messages:
                descriptions: list[str] = []
                if isinstance(message.content, list):
                    for part in message.content:
                        if isinstance(part, ImagePart):
                            if self.vision is None:
                                raise VisionUnavailable("vision API is not configured")
                            descriptions.append(
                                await self.vision.describe(
                                    part.image_url.url,
                                    context_text(message.content),
                                )
                            )
                indexed_texts.append(index_text(message.content, descriptions))
        except VisionError as error:
            self.database.mark_add_failed(request.request_id, "vision_failed")
            log_event(
                logger,
                logging.ERROR,
                "add_failed",
                request_id=request.request_id,
                stage="vision",
                error=type(error).__name__,
                duration_ms=duration_ms(started),
            )
            raise AddDependencyError("image description failed") from error
        episode_embedding_started = perf_counter()
        try:
            vectors = await self.embedder.embed(indexed_texts)
            if len(vectors) != len(request.messages):
                raise RuntimeError("embedding count mismatch")
        except Exception as error:
            self.database.mark_add_failed(request.request_id, "embedding_failed")
            log_event(
                logger,
                logging.ERROR,
                "add_failed",
                request_id=request.request_id,
                stage="episode_embedding",
                error=type(error).__name__,
                duration_ms=duration_ms(started),
            )
            raise AddDependencyError("episode embedding failed") from error
        log_event(
            logger,
            logging.INFO,
            "add_episode_embedding_completed",
            request_id=request.request_id,
            vectors=len(vectors),
            empty_vectors=sum(1 for vector in vectors if vector.size == 0),
            duration_ms=duration_ms(episode_embedding_started),
        )

        episodes = [
            EpisodeDraft(
                id=stable_id("episode", request.request_id, str(index)),
                user_id=request.user_id,
                session_id=request.session_id,
                request_id=request.request_id,
                message_index=index,
                role=message.role,
                content=indexed_texts[index],
                lexical_text=lexical_text(indexed_texts[index]),
                event_time=timestamp_to_datetime(message.timestamp),
                ingested_at=ingested_at,
                importance=1.0,
                embedding=vectors[index],
                original_content=(message.content if isinstance(message.content, list) else None),
            )
            for index, message in enumerate(request.messages)
        ]

        cards: list[MemoryCardDraft] = []
        state_versions = ()
        close_version_ids = ()
        raw_only = True
        degradation_stage = "extraction"
        extraction_started = perf_counter()
        try:
            extracted = await self.extractor.extract(
                [
                    MessageForExtraction(
                        index=index,
                        role=message.role,
                        content=indexed_texts[index],
                        timestamp=message.timestamp,
                    )
                    for index, message in enumerate(request.messages)
                ]
            )
            log_event(
                logger,
                logging.INFO,
                "add_extraction_completed",
                request_id=request.request_id,
                cards=len(extracted),
                duration_ms=duration_ms(extraction_started),
            )
            if extracted:
                card_texts = [
                    f"{card.subject} {card.predicate} {card.object}"
                    for card in extracted
                ]
                degradation_stage = "card_embedding"
                card_embedding_started = perf_counter()
                card_vectors = await self.embedder.embed(card_texts)
                if len(card_vectors) != len(extracted):
                    raise RuntimeError("card embedding count mismatch")
                log_event(
                    logger,
                    logging.INFO,
                    "add_card_embedding_completed",
                    request_id=request.request_id,
                    vectors=len(card_vectors),
                    empty_vectors=sum(
                        1 for vector in card_vectors if vector.size == 0
                    ),
                    duration_ms=duration_ms(card_embedding_started),
                )
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
        except ModelError as error:
            raw_only = True
            log_event(
                logger,
                logging.WARNING,
                "add_degraded",
                request_id=request.request_id,
                stage=degradation_stage,
                error=type(error).__name__,
            )
        except (KeyError, RuntimeError) as error:
            cards = []
            state_versions = ()
            close_version_ids = ()
            raw_only = True
            log_event(
                logger,
                logging.WARNING,
                "add_degraded",
                request_id=request.request_id,
                stage=degradation_stage,
                error=type(error).__name__,
            )
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "add_failed",
                request_id=request.request_id,
                stage=degradation_stage,
                error=type(error).__name__,
                duration_ms=duration_ms(started),
            )
            raise

        commit_started = perf_counter()
        try:
            self.database.commit_add(
                request_id=request.request_id,
                episodes=episodes,
                cards=cards,
                state_versions=state_versions,
                close_version_ids=close_version_ids,
                raw_only=raw_only,
            )
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "add_failed",
                request_id=request.request_id,
                stage="commit",
                error=type(error).__name__,
                duration_ms=duration_ms(started),
            )
            raise
        log_event(
            logger,
            logging.INFO,
            "add_commit_completed",
            request_id=request.request_id,
            episodes=len(episodes),
            cards=len(cards),
            raw_only=raw_only,
            duration_ms=duration_ms(commit_started),
        )
        log_event(
            logger,
            logging.INFO,
            "add_completed",
            request_id=request.request_id,
            replayed=False,
            duration_ms=duration_ms(started),
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
