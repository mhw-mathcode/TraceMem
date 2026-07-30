import asyncio

import httpx
import pytest

from tracemem.config import Settings
from tracemem.runtime import build_model_bundle


def _embedding_headers(extra_key: str) -> httpx.Headers:
    captured: list[httpx.Headers] = []

    def respond(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as client:
            settings = Settings(
                _env_file=None,
                embedding_mode="openai",
                embedding_api_key="bearer-secret",
                embedding_extra_key=extra_key,
                extraction_mode="disabled",
                rerank_mode="disabled",
            )
            models = build_model_bundle(settings, client)
            await models.embedder.embed(["hello"])

    asyncio.run(run())
    return captured[0]


def test_embedding_sends_both_credentials() -> None:
    headers = _embedding_headers("extra-secret")

    assert headers["Authorization"] == "Bearer bearer-secret"
    assert headers["X-Embedding-Key"] == "extra-secret"


@pytest.mark.parametrize("extra_key", ["", "   "])
def test_embedding_omits_blank_extra_credential(extra_key: str) -> None:
    headers = _embedding_headers(extra_key)

    assert headers["Authorization"] == "Bearer bearer-secret"
    assert "X-Embedding-Key" not in headers
