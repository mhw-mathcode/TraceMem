from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np
import pytest

from tracemem.add_service import AddService
from tracemem.api_models import AddRequest, MemoryMessage
from tracemem.db import Database
from tracemem.model_clients import MessageForExtraction, OpenAIExtractor
from tracemem.retrieval import HybridRetriever


class ConstantEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray([1.0, 0.0], dtype=np.float32) for _ in texts]


def _message(content: str = "I dislike cilantro") -> MessageForExtraction:
    return MessageForExtraction(
        index=0,
        role="user",
        content=content,
        timestamp=None,
    )


def _card(**overrides: Any) -> dict[str, Any]:
    card: dict[str, Any] = {
        "kind": "preference",
        "subject": "user",
        "predicate": "dislikes",
        "object": "cilantro",
        "event_time": None,
        "polarity": "negative",
        "confidence": 0.9,
        "importance": 0.8,
        "source_message_indexes": [0],
        "relation_hint": None,
    }
    card.update(overrides)
    return card


def _completion(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": json.dumps(payload),
                    "role": "assistant",
                },
            }
        ]
    }


def _extractor(client: httpx.AsyncClient) -> OpenAIExtractor:
    return OpenAIExtractor(
        client=client,
        url="https://llm.test/v1/chat/completions",
        api_key="test-key",
        model="test-model",
    )


@pytest.mark.asyncio
async def test_unknown_optional_relation_keeps_card(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response_payload = {
        "cards": [
            _card(
                kind=" Preference ",
                polarity=" Negative ",
                relation_hint="self-reported preference",
                object="SECRET_CARD_CONTENT",
            )
        ]
    }

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(response_payload))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        with caplog.at_level(logging.INFO, logger="tracemem.model_clients"):
            cards = await _extractor(client).extract(
                [_message("SECRET_MESSAGE_CONTENT")]
            )

    assert len(cards) == 1
    assert cards[0].kind == "preference"
    assert cards[0].polarity == "negative"
    assert cards[0].relation_hint is None
    assert (
        "event=card_extraction_parsed returned=1 accepted=1 "
        "normalized=1 rejected=0"
    ) in caplog.text
    assert "SECRET_CARD_CONTENT" not in caplog.text
    assert "SECRET_MESSAGE_CONTENT" not in caplog.text
    assert "self-reported preference" not in caplog.text


@pytest.mark.asyncio
async def test_invalid_cards_do_not_discard_valid_siblings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response_payload = {
        "cards": [
            _card(object="valid-memory"),
            _card(kind="unsupported-kind", object="SECRET_INVALID_KIND"),
            _card(
                source_message_indexes=[99],
                object="SECRET_INVALID_SOURCE",
            ),
        ]
    }

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(response_payload))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        with caplog.at_level(logging.INFO, logger="tracemem.model_clients"):
            cards = await _extractor(client).extract([_message()])

    assert [card.object for card in cards] == ["valid-memory"]
    assert (
        "event=card_extraction_parsed returned=3 accepted=1 "
        "normalized=0 rejected=2"
    ) in caplog.text
    assert "SECRET_INVALID_KIND" not in caplog.text
    assert "SECRET_INVALID_SOURCE" not in caplog.text


@pytest.mark.asyncio
async def test_extraction_request_enumerates_optional_contract_values() -> None:
    prompts: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        prompts.append(request_payload["messages"][1]["content"])
        return httpx.Response(200, json=_completion({"cards": []}))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        assert await _extractor(client).extract([_message()]) == []

    assert len(prompts) == 1
    assert (
        "polarity must be one of positive, negative, or uncertain"
        in prompts[0]
    )
    assert (
        "relation_hint must be one of continue, update, correction, "
        "conflict, or null"
        in prompts[0]
    )


@pytest.mark.asyncio
async def test_tolerated_card_is_committed_and_retrievable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    response_payload = {
        "cards": [
            _card(relation_hint="self-reported preference")
        ]
    }

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion(response_payload))

    database = Database(tmp_path / "cards.db")
    database.initialize()
    embedder = ConstantEmbedder()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        add_service = AddService(
            database=database,
            embedder=embedder,
            extractor=_extractor(client),
        )
        with caplog.at_level(logging.INFO, logger="tracemem"):
            response = await add_service.add(
                AddRequest(
                    request_id="card-integration",
                    user_id="user-1",
                    session_id="session-1",
                    messages=[
                        MemoryMessage(
                            role="user",
                            content="I dislike cilantro",
                        )
                    ],
                )
            )

    retriever = HybridRetriever(
        database=database,
        embedder=embedder,
        cards_enabled=True,
        state_enabled=False,
    )
    candidates = await retriever.retrieve(
        user_id="user-1",
        query="cilantro",
    )

    assert response.success is True
    assert database.count_cards("user-1") == 1
    assert any(
        item.source_type == "card" and "cilantro" in item.content
        for item in candidates
    )
    assert "cards=1 raw_only=false" in caplog.text
