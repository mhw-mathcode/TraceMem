from __future__ import annotations

from dataclasses import dataclass

import httpx

from tracemem.config import Settings
from tracemem.model_clients import (
    DashScopeReranker,
    DisabledExtractor,
    DisabledReranker,
    Embedder,
    Extractor,
    HashEmbedder,
    OpenAIEmbedder,
    OpenAIExtractor,
    OpenAIReranker,
    Reranker,
)
from tracemem.vision import OpenAIVisionDescriber, VisionDescriber


@dataclass(frozen=True)
class ModelBundle:
    embedder: Embedder
    extractor: Extractor
    reranker: Reranker
    vision: VisionDescriber


def build_model_bundle(
    settings: Settings,
    client: httpx.AsyncClient,
) -> ModelBundle:
    settings.require_remote_credentials()

    if settings.embedding_mode == "openai":
        embedder: Embedder = OpenAIEmbedder(
            client=client,
            url=settings.embedding_url,
            api_key=settings.embedding_api_key,
            extra_api_key=settings.embedding_extra_key,
            model=settings.embedding_model,
            timeout_seconds=settings.model_timeout_seconds,
            batch_size=settings.embedding_batch_size,
            max_concurrency=settings.embedding_max_concurrency,
        )
    else:
        embedder = HashEmbedder(dimensions=settings.embedding_dimensions)

    if settings.extraction_mode == "openai":
        extractor: Extractor = OpenAIExtractor(
            client=client,
            url=settings.llm_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        )
    else:
        extractor = DisabledExtractor()

    if settings.rerank_mode == "dashscope":
        reranker: Reranker = DashScopeReranker(
            client=client,
            url=settings.rerank_url,
            api_key=settings.llm_api_key,
            model=settings.rerank_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        )
    elif settings.rerank_mode == "openai":
        reranker = OpenAIReranker(
            client=client,
            url=settings.llm_url,
            api_key=settings.llm_api_key,
            model=settings.rerank_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        )
    else:
        reranker = DisabledReranker()

    return ModelBundle(
        embedder=embedder,
        extractor=extractor,
        reranker=reranker,
        vision=OpenAIVisionDescriber(
            client=client,
            url=settings.vision_url,
            api_key=settings.vision_api_key,
            model=settings.vision_model,
            timeout_seconds=settings.model_timeout_seconds,
            max_retries=settings.model_max_retries,
        ),
    )
