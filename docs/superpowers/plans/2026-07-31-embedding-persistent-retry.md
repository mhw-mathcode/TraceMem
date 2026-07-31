# Embedding Persistent Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make remote embedding calls retry temporary failures indefinitely with controlled backoff and a configurable concurrency limit.

**Architecture:** Add an embedding-specific retry loop to `OpenAIEmbedder` so the shared bounded retry helper remains unchanged for extraction and reranking. Limit individual outbound embedding attempts with an instance semaphore, release the permit during backoff, and propagate cancellation and permanent failures normally.

**Tech Stack:** Python 3.10+, asyncio, httpx, FastAPI settings, pytest

## Global Constraints

- Only `OpenAIEmbedder` receives persistent retry behavior.
- Retry HTTP 429, 500, 502, 503, and 504 plus connection, timeout, and premature TLS stream failures.
- Fail immediately for other HTTP status codes.
- Start exponential backoff at 0.5 seconds and cap it at 10 seconds.
- Honor valid `Retry-After` delta-seconds and HTTP-date values.
- Default embedding concurrency is three.
- Do not log credentials, request bodies, or response bodies.
- Do not catch task cancellation.

---

### Task 1: Persistent Embedding Retry Policy

**Files:**
- Modify: `src/tracemem/model_clients.py`
- Create: `tests/test_embedding_retry.py`

**Interfaces:**
- Consumes: `httpx.AsyncClient`, the existing `_RETRYABLE_STATUS_CODES`, and injected `Sleep`.
- Produces: `OpenAIEmbedder._post_until_success(headers, payload) -> httpx.Response`.

- [x] **Step 1: Write failing retry-policy tests**

Create async scenarios using `httpx.MockTransport` and an injected sleep
recorder. Assert that two 503 responses followed by 200 return an embedding,
that sleep delays are `[0.5, 1.0]`, that `Retry-After: 3` causes a three-second
delay, and that HTTP 401 raises `ModelTransportError` after one attempt.

```python
@pytest.mark.asyncio
async def test_embedding_retries_retryable_status_until_success() -> None:
    attempts = 0
    delays: list[float] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 0.0]}]},
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
        vectors = await embedder.embed(["hello"])

    assert attempts == 3
    assert delays == [0.5, 1.0]
    assert vectors[0].tolist() == [1.0, 0.0]
```

- [x] **Step 2: Run tests and verify the retry test fails**

Run: `pytest tests/test_embedding_retry.py -q`

Expected: FAIL because embedding stops after the existing bounded retry policy
or uses the old 0.25-second backoff.

- [x] **Step 3: Implement the embedding-specific retry loop**

Add helpers for retry delay parsing and a private method on `OpenAIEmbedder`.
The loop acquires the semaphore only for the HTTP exchange, retries temporary
failures forever, calls `raise_for_status()` for permanent responses, and lets
`asyncio.CancelledError` propagate.

```python
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
        return max(float(value), 0.0)
    except ValueError:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max((retry_at - current).total_seconds(), 0.0)


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
            if response.status_code not in _RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response
            delay = _retry_after_seconds(response)
            if delay is None:
                delay = _exponential_retry_delay(attempt)
            logger.warning(
                "Embedding retry attempt=%d status=%d delay=%.3f",
                attempt + 1,
                response.status_code,
                delay,
            )
        except (
            httpx.ConnectError,
            httpx.TimeoutException,
            anyio.EndOfStream,
        ) as error:
            delay = _exponential_retry_delay(attempt)
            logger.warning(
                "Embedding retry attempt=%d error=%s delay=%.3f",
                attempt + 1,
                type(error).__name__,
                delay,
            )
        attempt += 1
        await self.sleep(delay)
```

Replace the embedding call to `_post_with_retry` with
`self._post_until_success`. Preserve the existing conversion of terminal
`httpx.HTTPError` exceptions to `ModelTransportError`.

- [x] **Step 4: Run retry-policy and authentication tests**

Run: `pytest tests/test_embedding_retry.py tests/test_embedding_auth.py -q`

Expected: PASS.

### Task 2: Concurrency, Cancellation, and Configuration

**Files:**
- Modify: `tests/test_embedding_retry.py`
- Modify: `tests/test_embedding_auth.py`
- Modify: `src/tracemem/model_clients.py`
- Modify: `src/tracemem/config.py`
- Modify: `src/tracemem/runtime.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Settings.embedding_max_concurrency: int`.
- Produces: `OpenAIEmbedder(..., max_concurrency: int = 3)` and environment
  variable `TRACEMEM_EMBEDDING_MAX_CONCURRENCY`.

- [x] **Step 1: Write failing concurrency and cancellation tests**

Run four concurrent `embed()` calls against an async mock handler that tracks
active requests; block them on an event and assert no more than three enter the
handler. Add a retrying transport, cancel the embedding task during injected
sleep, assert `CancelledError`, and assert no later attempt occurs.

```python
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
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0]}]},
        )

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
        await asyncio.wait_for(three_active.wait(), timeout=1)
        assert maximum == 3
        release.set()
        await asyncio.gather(*tasks)

    assert maximum == 3


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
```

- [x] **Step 2: Run the focused tests and verify they fail**

Run: `pytest tests/test_embedding_retry.py -q`

Expected: FAIL because `max_concurrency` and the semaphore are not yet exposed.

- [x] **Step 3: Add concurrency configuration and documentation**

Add to `Settings`:

```python
embedding_max_concurrency: int = Field(default=3, ge=1, le=100)
```

Pass it from `build_model_bundle`:

```python
max_concurrency=settings.embedding_max_concurrency,
```

Initialize the embedder semaphore:

```python
self._semaphore = asyncio.Semaphore(max_concurrency)
```

Document this configuration in `.env.example` and `README.md`. Clarify that
`TRACEMEM_MODEL_MAX_RETRIES` applies to extraction and reranking while the
remote embedder persists on temporary failures.

- [x] **Step 4: Run all tests and static syntax validation**

Run: `pytest -q`

Expected: all tests PASS.

Run: `python -m compileall -q src tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: exit code 0.

- [ ] **Step 5: Commit the implementation**

```bash
git add src/tracemem/model_clients.py src/tracemem/config.py \
  src/tracemem/runtime.py tests/test_embedding_retry.py \
  tests/test_embedding_auth.py .env.example README.md \
  docs/superpowers/plans/2026-07-31-embedding-persistent-retry.md
git commit -m "feat: persistently retry embedding failures"
```
