from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from tracemem.add_service import (
    AddBusy,
    AddConflict,
    AddDependencyError,
    AddError,
    AddService,
)
from tracemem.api_models import (
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
)
from tracemem.auth import require_api_key
from tracemem.config import Settings
from tracemem.db import Database
from tracemem.evidence import EvidencePacker
from tracemem.model_clients import ModelError
from tracemem.observability import (
    configure_logging,
    duration_ms,
    log_event,
)
from tracemem.retrieval import HybridRetriever
from tracemem.runtime import build_model_bundle
from tracemem.search_service import SearchService


logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamped_database_path(
    path: Path,
    *,
    now: Callable[[], datetime] = _utc_now,
    process_id: int | None = None,
) -> Path:
    if path == Path(":memory:"):
        return path

    timestamp = now().astimezone(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    suffix = path.suffix
    base_name = path.stem if suffix else path.name
    primary = path.with_name(f"{base_name}-{timestamp}{suffix}")
    if not primary.exists():
        return primary

    pid = os.getpid() if process_id is None else process_id
    fallback = path.with_name(
        f"{base_name}-{timestamp}-p{pid}{suffix}"
    )
    counter = 2
    while fallback.exists():
        fallback = path.with_name(
            f"{base_name}-{timestamp}-p{pid}-{counter}{suffix}"
        )
        counter += 1
    return fallback


def _file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or Settings()
        configure_logging(active_settings.log_level)
        startup_started = perf_counter()
        stage = "settings_validation"
        log_event(
            logger,
            logging.INFO,
            "startup_started",
            profile=active_settings.profile,
            embedding_mode=active_settings.embedding_mode,
            extraction_mode=active_settings.extraction_mode,
            rerank_mode=active_settings.rerank_mode,
        )
        client: httpx.AsyncClient | None = None
        try:
            active_settings.require_api_key()
            app.state.api_key = active_settings.api_key

            stage = "database_initialization"
            database_started = perf_counter()
            database = Database(
                _timestamped_database_path(active_settings.database_path)
            )
            database.initialize()
            database_path = database.path.resolve()
            log_event(
                logger,
                logging.INFO,
                "database_initialized",
                path=database_path,
                size=_file_size(database_path),
                duration_ms=duration_ms(database_started),
            )

            stage = "model_initialization"
            client = httpx.AsyncClient()
            models = build_model_bundle(active_settings, client)
            retriever = HybridRetriever(
                database=database,
                embedder=models.embedder,
                cards_enabled=active_settings.cards_enabled,
                state_enabled=active_settings.temporal_enabled,
            )
            app.state.add_service = AddService(
                database=database,
                embedder=models.embedder,
                extractor=models.extractor,
                lease_seconds=active_settings.add_lease_seconds,
            )
            app.state.search_service = SearchService(
                retriever=retriever,
                final_result_limit=active_settings.final_result_limit,
                reranker=models.reranker,
                packer=EvidencePacker(
                    pairing_enabled=active_settings.evidence_pairing_enabled
                ),
            )
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "startup_failed",
                stage=stage,
                error=type(error).__name__,
                duration_ms=duration_ms(startup_started),
            )
            if client is not None:
                await client.aclose()
            raise

        log_event(
            logger,
            logging.INFO,
            "startup_completed",
            duration_ms=duration_ms(startup_started),
        )
        try:
            yield
        finally:
            shutdown_started = perf_counter()
            assert client is not None
            await client.aclose()
            log_event(
                logger,
                logging.INFO,
                "shutdown_completed",
                duration_ms=duration_ms(shutdown_started),
            )

    app = FastAPI(
        title="TraceMem",
        version="0.1.0",
        description="Temporal evidence-ledger memory service.",
        lifespan=lifespan,
    )

    @app.exception_handler(AddConflict)
    async def handle_add_conflict(
        _request: Request,
        error: AddConflict,
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(AddBusy)
    async def handle_add_busy(
        _request: Request,
        error: AddBusy,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(error)},
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(AddDependencyError)
    async def handle_add_dependency(
        _request: Request,
        error: AddDependencyError,
    ) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(ModelError)
    async def handle_model_error(
        _request: Request,
        error: ModelError,
    ) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(error)})

    @app.exception_handler(AddError)
    async def handle_add_error(
        _request: Request,
        error: AddError,
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(error)})

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/add",
        response_model=AddResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def add_memory(
        payload: AddRequest,
        request: Request,
    ) -> AddResponse:
        service: AddService = request.app.state.add_service
        return await service.add(payload)

    @app.post(
        "/search",
        response_model=SearchResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def search_memory(
        payload: SearchRequest,
        request: Request,
    ) -> SearchResponse:
        service: SearchService = request.app.state.search_service
        return await service.search(payload)

    return app


app = create_app()
