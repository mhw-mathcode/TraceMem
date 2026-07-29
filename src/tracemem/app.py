from __future__ import annotations

from contextlib import asynccontextmanager
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
from tracemem.retrieval import HybridRetriever
from tracemem.runtime import build_model_bundle
from tracemem.search_service import SearchService


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or Settings()
        active_settings.require_api_key()
        app.state.api_key = active_settings.api_key
        database = Database(active_settings.database_path)
        database.initialize()

        client = httpx.AsyncClient()
        try:
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
            yield
        finally:
            await client.aclose()

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
