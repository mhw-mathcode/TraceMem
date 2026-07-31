from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable, Protocol, Sequence

import anyio
import httpx
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from tracemem.domain import Candidate, ExtractedCard
from tracemem.text import lexical_text, timestamp_to_datetime


logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """Base error for an external or local model boundary."""


class ModelTransportError(ModelError):
    """The model endpoint could not complete a valid HTTP exchange."""


class ModelResponseError(ModelError):
    """The model endpoint returned a response that violates the contract."""


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
Sleep = Callable[[float], Awaitable[None]]


def _exponential_retry_delay(attempt: int) -> float:
    return min(0.5 * (2 ** min(attempt, 5)), 10.0)


def _retry_after_seconds(
    response: httpx.Response,
    *,
    now: datetime | None = None,
) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (OverflowError, TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max((retry_at - current).total_seconds(), 0.0)
    if not math.isfinite(delay) or delay < 0:
        return None
    return delay


def _openai_endpoint_url(url: str, endpoint: str) -> str:
    stripped = url.rstrip("/")
    if stripped.endswith(f"/{endpoint}"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/{endpoint}"
    return url


def _optional_timestamp(
    value: int | float | str | None,
):
    try:
        return timestamp_to_datetime(value)
    except (OSError, OverflowError, ValueError):
        return None


async def _post_with_retry(
    *,
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
    max_retries: int,
    sleep: Sleep,
) -> httpx.Response:
    for attempt in range(max_retries + 1):
        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds,
            )
            if (
                response.status_code in _RETRYABLE_STATUS_CODES
                and attempt < max_retries
            ):
                await sleep(0.25 * (2**attempt))
                continue
            response.raise_for_status()
            return response
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            anyio.EndOfStream,
        ) as error:
            if attempt >= max_retries:
                if isinstance(error, anyio.EndOfStream):
                    raise httpx.ConnectError(
                        "TLS stream ended unexpectedly"
                    ) from error
                raise
            await sleep(0.25 * (2**attempt))
    raise RuntimeError("retry loop exhausted")


@dataclass(frozen=True)
class MessageForExtraction:
    index: int
    role: str
    content: str
    timestamp: int | float | str | None


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]: ...


class Extractor(Protocol):
    async def extract(
        self,
        messages: Sequence[MessageForExtraction],
    ) -> list[ExtractedCard]: ...


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
    ) -> dict[str, float]: ...


class HashEmbedder:
    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in lexical_text(text).split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector


class DisabledExtractor:
    async def extract(
        self,
        messages: Sequence[MessageForExtraction],
    ) -> list[ExtractedCard]:
        return []


class DisabledReranker:
    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
    ) -> dict[str, float]:
        return {candidate.id: candidate.score for candidate in candidates}


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        api_key: str,
        extra_api_key: str = "",
        model: str,
        timeout_seconds: float = 30.0,
        batch_size: int = 10,
        max_concurrency: int = 3,
        max_retries: int = 2,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.client = client
        self.url = _openai_endpoint_url(url, "embeddings")
        self.api_key = api_key
        self.extra_api_key = extra_api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Retained in the constructor for compatibility with existing callers.
        # Remote embedding retries are intentionally no longer count-limited.
        _ = max_retries
        self.sleep = sleep

    async def _post_until_success(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        attempt = 0
        while True:
            try:
                async with self._semaphore:
                    response = await self.client.post(
                        self.url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout_seconds,
                    )
            except (httpx.TransportError, anyio.EndOfStream) as error:
                delay = _exponential_retry_delay(attempt)
                logger.warning(
                    "Embedding retry attempt=%d error=%s delay=%.3f",
                    attempt + 1,
                    type(error).__name__,
                    delay,
                )
            else:
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    return response
                retry_after = _retry_after_seconds(response)
                delay = (
                    retry_after
                    if retry_after is not None
                    else _exponential_retry_delay(attempt)
                )
                logger.warning(
                    "Embedding retry attempt=%d status=%d delay=%.3f",
                    attempt + 1,
                    response.status_code,
                    delay,
                )
            attempt += 1
            await self.sleep(delay)

    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors: list[np.ndarray] = []
        expected_dimensions: int | None = None
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset : offset + self.batch_size])
            headers = {"Authorization": f"Bearer {self.api_key}"}
            if self.extra_api_key.strip():
                headers["X-Embedding-Key"] = self.extra_api_key.strip()
            try:
                response = await self._post_until_success(
                    headers=headers,
                    payload={"model": self.model, "input": batch},
                )
            except httpx.HTTPError as error:
                raise ModelTransportError(
                    f"Embedding endpoint failed with {type(error).__name__}"
                ) from error
            try:
                items = sorted(
                    response.json()["data"],
                    key=lambda item: item["index"],
                )
                if len(items) != len(batch):
                    raise ValueError("embedding count mismatch")
                batch_vectors = [
                    np.asarray(item["embedding"], dtype=np.float32).reshape(-1)
                    for item in items
                ]
                if not batch_vectors:
                    raise ValueError("empty embedding response")
                if expected_dimensions is None:
                    expected_dimensions = batch_vectors[0].size
                if expected_dimensions == 0 or any(
                    vector.size != expected_dimensions
                    for vector in batch_vectors
                ):
                    raise ValueError("embedding dimensions mismatch")
                vectors.extend(batch_vectors)
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise ModelResponseError(
                    "Embedding response has an invalid schema"
                ) from error
        return vectors


class _CardPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    event_time: int | float | str | None = None
    polarity: str
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(ge=0, le=1)
    source_message_indexes: list[int] = Field(min_length=1)
    relation_hint: str | None = None

    @field_validator("confidence", "importance", mode="before")
    @classmethod
    def normalize_named_score(cls, value):
        if isinstance(value, str):
            named = {"low": 0.3, "medium": 0.6, "high": 0.9}
            normalized = value.strip().casefold()
            if normalized in named:
                return named[normalized]
        return value


class _ExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cards: list[_CardPayload]


class OpenAIExtractor:
    _KINDS = {
        "fact",
        "event",
        "preference",
        "goal",
        "plan",
        "relationship",
        "causal",
    }
    _POLARITIES = {"positive", "negative", "uncertain"}
    _RELATIONS = {None, "continue", "update", "correction", "conflict"}

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.client = client
        self.url = _openai_endpoint_url(url, "chat/completions")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep = sleep

    async def extract(
        self,
        messages: Sequence[MessageForExtraction],
    ) -> list[ExtractedCard]:
        if not messages:
            return []
        allowed_indexes = {message.index for message in messages}
        input_payload = [
            {
                "index": message.index,
                "role": message.role,
                "timestamp": message.timestamp,
                "content": message.content,
            }
            for message in messages
        ]
        prompt = (
            "Extract only durable, future-useful memories. Return strict JSON with "
            'top-level key "cards". Every card needs kind, subject, predicate, object, '
            "event_time, polarity, confidence, importance, source_message_indexes, "
            "and relation_hint. kind must be one of fact, event, preference, goal, "
            "plan, relationship, or causal. event_time must be ISO-8601 or null, "
            "never relative text. confidence and importance must be numbers from "
            "0 to 1. Do not infer unsupported facts.\n\n"
            + json.dumps(input_payload, ensure_ascii=False)
        )
        try:
            response = await _post_with_retry(
                client=self.client,
                url=self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                payload={
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": "You extract traceable agent memories.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
                sleep=self.sleep,
            )
        except httpx.HTTPError as error:
            raise ModelTransportError(
                f"Extraction endpoint failed with {type(error).__name__}"
            ) from error
        try:
            content = response.json()["choices"][0]["message"]["content"]
            payload = _ExtractionPayload.model_validate_json(content)
        except (
            KeyError,
            IndexError,
            TypeError,
            ValidationError,
            json.JSONDecodeError,
        ) as error:
            raise ModelResponseError("Extraction response has an invalid schema") from error

        extracted: list[ExtractedCard] = []
        for card in payload.cards:
            kind = "fact" if card.kind == "memory" else card.kind
            if kind not in self._KINDS:
                raise ModelResponseError(f"Unsupported memory kind: {kind}")
            if card.polarity not in self._POLARITIES:
                raise ModelResponseError(f"Unsupported polarity: {card.polarity}")
            if card.relation_hint not in self._RELATIONS:
                raise ModelResponseError(
                    f"Unsupported relation hint: {card.relation_hint}"
                )
            if not set(card.source_message_indexes) <= allowed_indexes:
                raise ModelResponseError("Card references an unknown source message index")
            extracted.append(
                ExtractedCard(
                    kind=kind,  # type: ignore[arg-type]
                    subject=card.subject,
                    predicate=card.predicate,
                    object=card.object,
                    event_time=_optional_timestamp(card.event_time),
                    polarity=card.polarity,  # type: ignore[arg-type]
                    confidence=card.confidence,
                    importance=card.importance,
                    source_message_indexes=tuple(card.source_message_indexes),
                    relation_hint=card.relation_hint,  # type: ignore[arg-type]
                )
            )
        return extracted


class OpenAIReranker:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.client = client
        self.url = _openai_endpoint_url(url, "chat/completions")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep = sleep

    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
    ) -> dict[str, float]:
        if not candidates:
            return {}
        compact = [
            {"id": candidate.id, "content": candidate.content[:2000]}
            for candidate in candidates
        ]
        try:
            response = await _post_with_retry(
                client=self.client,
                url=self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                payload={
                    "model": self.model,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Score each memory from 0 to 1 for answering the query. "
                                'Return {"scores":[{"id":"...","score":0.0}]}.\n'
                                f"Query: {query}\nMemories: "
                                + json.dumps(compact, ensure_ascii=False)
                            ),
                        }
                    ],
                },
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
                sleep=self.sleep,
            )
            payload: dict[str, Any] = response.json()
            content = json.loads(payload["choices"][0]["message"]["content"])
            scores = {
                str(item["id"]): min(1.0, max(0.0, float(item["score"])))
                for item in content["scores"]
            }
            return scores
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            if isinstance(error, httpx.HTTPError):
                raise ModelTransportError(
                    f"Rerank endpoint failed with {type(error).__name__}"
                ) from error
            raise ModelResponseError("Rerank response has an invalid schema") from error


class DashScopeReranker:
    _INSTRUCT = (
        "Given a memory question, retrieve passages that contain direct "
        "or supporting evidence needed to answer it."
    )

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.client = client
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep = sleep

    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
    ) -> dict[str, float]:
        if not candidates:
            return {}
        try:
            response = await _post_with_retry(
                client=self.client,
                url=self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                payload={
                    "model": self.model,
                    "query": query,
                    "documents": [
                        candidate.content[:8000] for candidate in candidates
                    ],
                    "top_n": len(candidates),
                    "instruct": self._INSTRUCT,
                },
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
                sleep=self.sleep,
            )
        except httpx.HTTPError as error:
            raise ModelTransportError(
                f"Rerank endpoint failed with {type(error).__name__}"
            ) from error

        try:
            results = response.json()["output"]["results"]
            if not isinstance(results, list):
                raise ValueError("results must be a list")
            scores: dict[str, float] = {}
            seen_indexes: set[int] = set()
            for result in results:
                index = result["index"]
                raw_score = result["relevance_score"]
                if type(index) is not int or not 0 <= index < len(candidates):
                    raise ValueError("result index out of range")
                if index in seen_indexes:
                    raise ValueError("duplicate result index")
                if isinstance(raw_score, bool):
                    raise ValueError("score must be numeric")
                score = float(raw_score)
                if not math.isfinite(score):
                    raise ValueError("score must be finite")
                seen_indexes.add(index)
                scores[candidates[index].id] = min(1.0, max(0.0, score))
            return scores
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ModelResponseError(
                "Rerank response has an invalid schema"
            ) from error
