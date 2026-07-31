from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pytest

from tracemem.add_service import AddService
from tracemem.api_models import (
    AddRequest,
    MemoryMessage,
    SearchRequest,
)
from tracemem.app import create_app
from tracemem.config import Settings
from tracemem.db import Database
from tracemem.domain import Candidate
from tracemem.model_clients import (
    DisabledExtractor,
    MessageForExtraction,
    ModelTransportError,
)
from tracemem.observability import (
    configure_logging,
    duration_ms,
    log_event,
)
from tracemem.retrieval import HybridRetriever
from tracemem.search_service import SearchService


class FakeEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [
            np.asarray([1.0, float(index + 1)], dtype=np.float32)
            for index, _text in enumerate(texts)
        ]


class FailingExtractor:
    async def extract(
        self,
        messages: Sequence[MessageForExtraction],
    ) -> list[object]:
        raise ModelTransportError("SECRET_REMOTE_RESPONSE")


class FakeRetriever:
    def __init__(self) -> None:
        self.search_id: str | None = None

    async def retrieve(
        self,
        *,
        user_id: str,
        query: str,
        options: Sequence[str] | None = None,
        plan: object | None = None,
        search_id: str | None = None,
    ) -> list[Candidate]:
        self.search_id = search_id
        return [
            Candidate(
                id="candidate-1",
                source_type="episode",
                content="SECRET_CANDIDATE_CONTENT",
                created_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
                score=0.8,
            )
        ]


class FailingReranker:
    async def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
    ) -> dict[str, float]:
        raise ModelTransportError("SECRET_RERANK_BODY")


@pytest.fixture(autouse=True)
def reset_tracemem_logger() -> Iterator[None]:
    logger = logging.getLogger("tracemem")
    original_level = logger.level
    original_propagate = logger.propagate
    original_handlers = list(logger.handlers)
    yield
    logger.handlers[:] = original_handlers
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_configure_logging_is_idempotent() -> None:
    first_stream = StringIO()
    second_stream = StringIO()

    logger = configure_logging("INFO", stream=first_stream)
    again = configure_logging("DEBUG", stream=second_stream)

    tagged = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_tracemem_handler", False)
    ]
    assert again is logger
    assert logger.level == logging.DEBUG
    assert len(tagged) == 1

    log_event(logger, logging.INFO, "probe", value="second")
    assert first_stream.getvalue() == ""
    assert "event=probe value=second" in second_stream.getvalue()


def test_log_event_escapes_newlines() -> None:
    stream = StringIO()
    logger = configure_logging("INFO", stream=stream)

    log_event(
        logger,
        logging.INFO,
        "probe",
        request_id="ok\nforged=true",
    )

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert "ok\\nforged=true" in output


def test_log_event_rejects_non_scalar_fields() -> None:
    stream = StringIO()
    logger = configure_logging("INFO", stream=stream)

    with pytest.raises(TypeError, match="scalar"):
        log_event(
            logger,
            logging.INFO,
            "probe",
            payload={"message": "secret"},  # type: ignore[arg-type]
        )


def test_log_event_renders_paths_and_booleans() -> None:
    stream = StringIO()
    logger = configure_logging("INFO", stream=stream)

    log_event(
        logger,
        logging.INFO,
        "database_initialized",
        path=Path("memory.db"),
        ready=True,
    )

    assert "path=memory.db ready=true" in stream.getvalue()


def test_duration_ms_uses_monotonic_difference() -> None:
    assert duration_ms(10.0, now=lambda: 10.125) == 125


def test_log_level_defaults_to_info() -> None:
    assert Settings(_env_file=None).log_level == "INFO"


@pytest.mark.asyncio
async def test_lifespan_logs_database_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    database_path = tmp_path / "runtime.db"

    def configure_for_test(level: str) -> logging.Logger:
        return configure_logging(level, stream=stream)

    monkeypatch.setattr(
        "tracemem.app.configure_logging",
        configure_for_test,
        raising=False,
    )
    settings = Settings(
        _env_file=None,
        api_key="service-secret",
        database_path=database_path,
        embedding_mode="hash",
        extraction_mode="disabled",
        rerank_mode="disabled",
    )

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        pass

    assert database_path.stat().st_size > 0
    output = stream.getvalue()
    assert "event=startup_started" in output
    assert "event=database_initialized" in output
    assert "runtime.db" in output
    assert "event=startup_completed" in output
    assert "event=shutdown_completed" in output


@pytest.mark.asyncio
async def test_lifespan_logging_does_not_require_database_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()

    def configure_for_test(level: str) -> logging.Logger:
        return configure_logging(level, stream=stream)

    monkeypatch.setattr(
        "tracemem.app.configure_logging",
        configure_for_test,
    )
    settings = Settings(
        _env_file=None,
        api_key="service-secret",
        database_path=Path(":memory:"),
        embedding_mode="hash",
        extraction_mode="disabled",
        rerank_mode="disabled",
    )
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        pass

    output = stream.getvalue()
    assert "event=database_initialized" in output
    assert "size=null" in output
    assert "event=startup_completed" in output


@pytest.mark.asyncio
async def test_add_logs_stages_without_content_or_identity(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = Database(tmp_path / "memory.db")
    database.initialize()
    request = AddRequest(
        request_id="req-safe",
        user_id="SECRET_USER_ID",
        session_id="SECRET_SESSION_ID",
        messages=[
            MemoryMessage(role="user", content="SECRET_MESSAGE_BODY")
        ],
    )
    service = AddService(
        database=database,
        embedder=FakeEmbedder(),
        extractor=DisabledExtractor(),
    )

    with caplog.at_level(logging.INFO, logger="tracemem.add_service"):
        response = await service.add(request)

    assert response.success is True
    assert "event=add_started request_id=req-safe messages=1" in caplog.text
    assert "event=add_claimed" in caplog.text
    assert "event=add_episode_embedding_completed" in caplog.text
    assert "event=add_extraction_completed" in caplog.text
    assert "event=add_commit_completed" in caplog.text
    assert "event=add_completed" in caplog.text
    for secret in (
        "SECRET_USER_ID",
        "SECRET_SESSION_ID",
        "SECRET_MESSAGE_BODY",
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_add_logs_safe_extraction_degradation(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = Database(tmp_path / "degraded.db")
    database.initialize()
    request = AddRequest(
        request_id="req-degraded",
        user_id="user",
        session_id="session",
        messages=[MemoryMessage(role="user", content="safe input")],
    )
    service = AddService(
        database=database,
        embedder=FakeEmbedder(),
        extractor=FailingExtractor(),  # type: ignore[arg-type]
    )

    with caplog.at_level(logging.INFO, logger="tracemem.add_service"):
        response = await service.add(request)

    assert response.success is True
    assert (
        "event=add_degraded request_id=req-degraded stage=extraction "
        "error=ModelTransportError"
    ) in caplog.text
    assert "raw_only=true" in caplog.text
    assert "SECRET_REMOTE_RESPONSE" not in caplog.text


@pytest.mark.asyncio
async def test_add_logs_idempotent_replay(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = Database(tmp_path / "replay.db")
    database.initialize()
    request = AddRequest(
        request_id="req-replay",
        user_id="user",
        session_id="session",
        messages=[MemoryMessage(role="user", content="safe input")],
    )
    service = AddService(
        database=database,
        embedder=FakeEmbedder(),
        extractor=DisabledExtractor(),
    )
    await service.add(request)
    caplog.clear()

    with caplog.at_level(logging.INFO, logger="tracemem.add_service"):
        response = await service.add(request)

    assert response.success is True
    assert (
        "event=add_completed request_id=req-replay replayed=true"
    ) in caplog.text


@pytest.mark.asyncio
async def test_search_logs_correlated_stages_without_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    retriever = FakeRetriever()
    service = SearchService(retriever=retriever)  # type: ignore[arg-type]
    request = SearchRequest(
        query="SECRET_QUERY",
        options=["SECRET_OPTION"],
        user_id="SECRET_SEARCH_USER",
        top_k=5,
    )

    with caplog.at_level(logging.INFO, logger="tracemem.search_service"):
        response = await service.search(request)

    assert len(response.data) == 1
    assert "event=search_started" in caplog.text
    assert "event=search_retrieved" in caplog.text
    assert "event=search_reranked" in caplog.text
    assert "event=search_completed" in caplog.text
    ids = re.findall(r"search_id=([0-9a-f]{12})", caplog.text)
    assert len(ids) >= 4
    assert len(set(ids)) == 1
    assert retriever.search_id == ids[0]
    for secret in (
        "SECRET_QUERY",
        "SECRET_OPTION",
        "SECRET_SEARCH_USER",
        "SECRET_CANDIDATE_CONTENT",
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_search_logs_safe_rerank_degradation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    retriever = FakeRetriever()
    service = SearchService(
        retriever=retriever,  # type: ignore[arg-type]
        reranker=FailingReranker(),
    )
    request = SearchRequest(
        query="safe query",
        user_id="safe-user",
        top_k=5,
    )

    with caplog.at_level(logging.INFO, logger="tracemem.search_service"):
        response = await service.search(request)

    assert len(response.data) == 1
    assert "event=search_degraded" in caplog.text
    assert "stage=rerank error=ModelTransportError" in caplog.text
    assert "SECRET_RERANK_BODY" not in caplog.text


@pytest.mark.asyncio
async def test_retriever_logs_embedding_and_channel_counts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = Database(tmp_path / "retrieval.db")
    database.initialize()
    retriever = HybridRetriever(
        database=database,
        embedder=FakeEmbedder(),
        cards_enabled=False,
        state_enabled=False,
    )

    with caplog.at_level(logging.INFO, logger="tracemem.retrieval"):
        candidates = await retriever.retrieve(
            user_id="SECRET_RETRIEVAL_USER",
            query="SECRET_RETRIEVAL_QUERY",
            options=["SECRET_RETRIEVAL_OPTION"],
            search_id="abcdef123456",
        )

    assert candidates == []
    assert (
        "event=search_query_embedding_completed "
        "search_id=abcdef123456 vectors=1"
    ) in caplog.text
    assert "event=search_channels_completed search_id=abcdef123456" in caplog.text
    assert "episode_bm25=0" in caplog.text
    assert "episode_vector=0" in caplog.text
    for secret in (
        "SECRET_RETRIEVAL_USER",
        "SECRET_RETRIEVAL_QUERY",
        "SECRET_RETRIEVAL_OPTION",
    ):
        assert secret not in caplog.text
