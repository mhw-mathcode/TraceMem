import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import pytest

from tracemem.config import Settings
from tracemem.model_clients import OpenAIEmbedder
from tracemem.runtime import build_model_bundle


def _embedding_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
    )


@pytest.mark.asyncio
async def test_embedding_retries_retryable_status_until_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 5:
            return httpx.Response(503)
        return _embedding_response()

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://SECRET_ENDPOINT.test/v1/embeddings",
            api_key="SECRET_API_KEY",
            model="embedding",
            max_retries=2,
            sleep=record_sleep,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="tracemem.model_clients",
        ):
            vectors = await embedder.embed(["SECRET_EMBEDDING_INPUT"])

    assert attempts == 5
    assert delays == [0.5, 1.0, 2.0, 4.0]
    assert vectors[0].tolist() == [1.0, 0.0]
    assert "event=embedding_retry attempt=1 status=503 delay=0.5" in caplog.text
    for secret in (
        "SECRET_ENDPOINT",
        "SECRET_API_KEY",
        "SECRET_EMBEDDING_INPUT",
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_embedding_honors_retry_after_delta_seconds() -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return _embedding_response()

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        await embedder.embed(["hello"])

    assert attempts == 2
    assert delays == [3.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retry_after",
    ["NaN", "Infinity", "-Infinity", "-1"],
)
async def test_embedding_ignores_non_finite_retry_after(
    retry_after: str,
) -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                headers={"Retry-After": retry_after},
            )
        return _embedding_response()

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        await embedder.embed(["hello"])

    assert attempts == 2
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_embedding_degrades_permanent_http_error_after_ten_attempts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(401)

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://SECRET_ENDPOINT.test/v1/embeddings",
            api_key="SECRET_API_KEY",
            model="embedding",
            sleep=record_sleep,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="tracemem.model_clients",
        ):
            vectors = await embedder.embed(["SECRET_INPUT"])

    assert attempts == 10
    assert delays == [0.5] * 9
    assert len(vectors) == 1
    assert vectors[0].size == 0
    assert "status=401" in caplog.text
    assert "attempt=10" in caplog.text
    assert "batch_items=1" in caplog.text
    assert "max_chars=12" in caplog.text
    assert "total_chars=12" in caplog.text
    for secret in (
        "SECRET_ENDPOINT",
        "SECRET_API_KEY",
        "SECRET_INPUT",
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_embedding_recovers_from_permanent_error_before_limit() -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(400)
        return _embedding_response()

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        vectors = await embedder.embed(["hello"])

    assert attempts == 3
    assert delays == [0.5, 0.5]
    assert vectors[0].tolist() == [1.0, 0.0]


@pytest.mark.asyncio
async def test_embedding_isolates_batch_and_preserves_good_items() -> None:
    batch_attempts = 0
    isolated_inputs: list[str] = []
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal batch_attempts
        inputs = json.loads(request.content)["input"]
        if len(inputs) == 2:
            batch_attempts += 1
            return httpx.Response(400)
        isolated_inputs.append(inputs[0])
        if inputs[0] == "bad":
            return httpx.Response(422)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
        )

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        vectors = await embedder.embed(["good", "bad"])

    assert batch_attempts == 10
    assert isolated_inputs == ["good", "bad"]
    assert delays == [0.5] * 9
    assert vectors[0].tolist() == [1.0, 2.0]
    assert vectors[1].size == 0


@pytest.mark.asyncio
async def test_embedding_degrades_invalid_success_response() -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"data": []})

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        vectors = await embedder.embed(["hello"])

    assert attempts == 10
    assert delays == [0.5] * 9
    assert len(vectors) == 1
    assert vectors[0].size == 0


@pytest.mark.asyncio
async def test_embedding_rejects_non_positive_concurrency() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="max_concurrency"):
            OpenAIEmbedder(
                client=client,
                url="https://example.test/v1/embeddings",
                api_key="secret",
                model="embedding",
                max_concurrency=0,
            )


@pytest.mark.asyncio
async def test_embedding_caps_exponential_backoff_at_ten_seconds() -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 8:
            return httpx.Response(503)
        return _embedding_response()

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        await embedder.embed(["hello"])

    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


@pytest.mark.asyncio
async def test_embedding_honors_retry_after_http_date() -> None:
    attempts = 0
    delays: list[float] = []
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=60)

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                503,
                headers={"Retry-After": format_datetime(retry_at)},
            )
        return _embedding_response()

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        await embedder.embed(["hello"])

    assert len(delays) == 1
    assert 55.0 <= delays[0] <= 60.0


@pytest.mark.asyncio
async def test_embedding_recovers_from_connection_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("temporary failure", request=request)
        return _embedding_response()

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=record_sleep,
        )
        with caplog.at_level(
            logging.WARNING,
            logger="tracemem.model_clients",
        ):
            await embedder.embed(["hello"])

    assert attempts == 2
    assert delays == [0.5]
    assert (
        "event=embedding_retry attempt=1 error=ConnectError delay=0.5"
    ) in caplog.text


@pytest.mark.asyncio
async def test_embedding_limits_concurrent_http_attempts() -> None:
    active = 0
    maximum = 0
    lock = asyncio.Lock()
    three_active = asyncio.Event()
    release = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        async with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 3:
                three_active.set()
        await release.wait()
        async with lock:
            active -= 1
        return _embedding_response()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            max_concurrency=3,
        )
        tasks = [
            asyncio.create_task(embedder.embed([str(index)]))
            for index in range(4)
        ]
        try:
            await asyncio.wait_for(three_active.wait(), timeout=1)
            observed_maximum = maximum
        finally:
            release.set()
            await asyncio.gather(*tasks)

    assert observed_maximum == 3


@pytest.mark.asyncio
async def test_embedding_propagates_cancellation_during_backoff() -> None:
    attempts = 0
    sleeping = asyncio.Event()
    never = asyncio.Event()

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    async def blocked_sleep(delay: float) -> None:
        sleeping.set()
        await never.wait()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://example.test/v1/embeddings",
            api_key="secret",
            model="embedding",
            sleep=blocked_sleep,
        )
        task = asyncio.create_task(embedder.embed(["hello"]))
        await asyncio.wait_for(sleeping.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert attempts == 1


@pytest.mark.asyncio
async def test_runtime_applies_embedding_concurrency_setting() -> None:
    active = 0
    maximum = 0
    first_active = asyncio.Event()
    release = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        first_active.set()
        await release.wait()
        active -= 1
        return _embedding_response()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond)
    ) as client:
        settings = Settings(
            _env_file=None,
            embedding_mode="openai",
            embedding_api_key="secret",
            embedding_max_concurrency=1,
            extraction_mode="disabled",
            rerank_mode="disabled",
        )
        embedder = build_model_bundle(settings, client).embedder
        tasks = [
            asyncio.create_task(embedder.embed([str(index)]))
            for index in range(2)
        ]
        try:
            await asyncio.wait_for(first_active.wait(), timeout=1)
            await asyncio.sleep(0.05)
            observed_maximum = maximum
        finally:
            release.set()
            await asyncio.gather(*tasks)

    assert observed_maximum == 1
