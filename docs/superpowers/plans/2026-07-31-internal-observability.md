# Internal Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe structured internal logs for startup, Add, Search, retrieval, degradation, and embedding retries.

**Architecture:** A focused `observability.py` module configures the `tracemem` logger, renders safe scalar fields, and measures monotonic durations. Existing services emit explicit allowlisted fields at their stage boundaries; no request/model object is passed into logging helpers.

**Tech Stack:** Python 3.10+, standard-library logging, FastAPI lifespan, pytest, asyncio

## Global Constraints

- Write TraceMem logs to standard error; do not open or rotate application log files.
- `TRACEMEM_LOG_LEVEL` defaults to `INFO` and accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
- Do not log message/query/options/card content, user/session IDs, credentials, headers, model bodies, or remote exception messages.
- Use stable `event=... key=value` messages and monotonic integer millisecond durations.
- Keep all public HTTP request and response contracts unchanged.
- Preserve existing Add/Search success, error, retry, and degradation behavior.

---

### Task 1: Logging Core, Configuration, and Startup

**Files:**
- Create: `src/tracemem/observability.py`
- Create: `tests/test_observability.py`
- Modify: `src/tracemem/config.py`
- Modify: `src/tracemem/app.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Produces: `configure_logging(level: str, *, stream: TextIO | None = None) -> logging.Logger`.
- Produces: `log_event(logger, level, event, **fields) -> None` where fields are safe scalars.
- Produces: `duration_ms(started: float, *, now: Callable[[], float] = perf_counter) -> int`.
- Produces: `Settings.log_level` from `TRACEMEM_LOG_LEVEL`.

- [ ] **Step 1: Write failing logging-core tests**

Test idempotent configuration, level selection, log-line injection resistance,
unsupported field rejection, and literal duration calculation.

```python
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
    stream = StringIO()
    logger = configure_logging("INFO", stream=stream)
    again = configure_logging("DEBUG", stream=stream)

    tagged = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_tracemem_handler", False)
    ]
    assert again is logger
    assert logger.level == logging.DEBUG
    assert len(tagged) == 1


def test_log_event_escapes_newlines() -> None:
    stream = StringIO()
    logger = configure_logging("INFO", stream=stream)
    log_event(logger, logging.INFO, "probe", request_id="ok\nforged=true")

    output = stream.getvalue()
    assert output.count("\n") == 1
    assert "ok\\nforged=true" in output


def test_duration_ms_uses_monotonic_difference() -> None:
    assert duration_ms(10.0, now=lambda: 10.125) == 125
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest tests/test_observability.py -q`

Expected: import failure because `tracemem.observability` does not exist.

- [ ] **Step 3: Implement the logging core and setting**

Create `observability.py` with an allowlisted scalar renderer. Safe tokens remain
unquoted; other strings use JSON encoding so newlines become `\\n`.

```python
Scalar = str | int | float | bool | None | Path | Enum
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._:/-]+$")


def _render(value: Scalar) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        value = str(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        raise TypeError("log fields must be scalar values")
    return value if _SAFE_TOKEN.fullmatch(value) else json.dumps(value)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Scalar,
) -> None:
    parts = [f"event={_render(event)}"]
    parts.extend(f"{key}={_render(value)}" for key, value in fields.items())
    logger.log(level, " ".join(parts))
```

`configure_logging` adds one marked `StreamHandler`, updates its stream when a
test supplies one, applies the requested level, and sets `propagate=False`.
Add `log_level: Literal[...] = "INFO"` to `Settings`.

- [ ] **Step 4: Write and run failing startup event test**

Use `TestClient(create_app(Settings(...)))` with local hash/disabled models and
a temporary database. Monkeypatch `tracemem.app.configure_logging` to pass a
test-owned `StringIO` stream. Require
`startup_started`, `database_initialized`, and `startup_completed`; assert the
resolved database path is present and the file size becomes nonzero.

Run: `pytest tests/test_observability.py::test_lifespan_logs_database_initialization -q`

Expected: FAIL because lifespan emits no internal events.

- [ ] **Step 5: Add startup events and documentation**

In the lifespan, configure logging before initialization, track `stage`, and
emit only mode names, resolved path, size, and durations. On failure emit the
exception class and re-raise. Add `TRACEMEM_LOG_LEVEL=INFO` to `.env.example`
and document standard-error collection in README.

- [ ] **Step 6: Verify Task 1**

Run: `pytest tests/test_observability.py -q`

Expected: PASS.

### Task 2: Add Stage Logs and Privacy

**Files:**
- Modify: `src/tracemem/add_service.py`
- Modify: `tests/test_observability.py`

**Interfaces:**
- Consumes: `log_event`, `duration_ms`, module logger `tracemem.add_service`.
- Produces: Add events defined in the design, correlated only by `request_id`.

- [ ] **Step 1: Write failing successful-Add logging test**

Use a real temporary `Database`, a deterministic fake embedder, and
`DisabledExtractor`. Send content, user ID, and session ID containing unique
secret markers. Require start, episode embedding, extraction, commit, and
completion events with counts; assert every secret marker is absent.

```python
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
        messages=[MemoryMessage(role="user", content="SECRET_MESSAGE_BODY")],
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
    assert "event=add_episode_embedding_completed" in caplog.text
    assert "event=add_commit_completed" in caplog.text
    assert "event=add_completed" in caplog.text
    for secret in ("SECRET_USER_ID", "SECRET_SESSION_ID", "SECRET_MESSAGE_BODY"):
        assert secret not in caplog.text
```

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/test_observability.py::test_add_logs_stages_without_content_or_identity -q`

Expected: FAIL because Add emits no events.

- [ ] **Step 3: Add stage timers and events**

Instrument claim, episode embedding, extraction, card embedding, commit, and
overall completion. Track the current `stage`; log only `type(error).__name__`
for failure/degradation. Preserve the existing exception conversions and
raw-only fallback.

```python
started = perf_counter()
stage = "claim"
log_event(logger, logging.INFO, "add_started", request_id=request.request_id,
          messages=len(request.messages))

embedding_started = perf_counter()
vectors = await self.embedder.embed([...])
log_event(
    logger,
    logging.INFO,
    "add_episode_embedding_completed",
    request_id=request.request_id,
    vectors=len(vectors),
    duration_ms=duration_ms(embedding_started),
)
```

- [ ] **Step 4: Write RED degradation and replay tests**

Use an extractor that raises `ModelTransportError("SECRET_REMOTE_RESPONSE")`;
require `event=add_degraded stage=extraction error=ModelTransportError`, assert
the remote message is absent, and verify raw-only persistence still succeeds.
Call the same request twice and require the second `add_completed` event to have
`replayed=true`.

- [ ] **Step 5: Implement degradation/replay events and verify Task 2**

Run: `pytest tests/test_observability.py -q`

Expected: PASS with unchanged Add responses and persistence behavior.

### Task 3: Search, Retrieval, and Retry Events

**Files:**
- Modify: `src/tracemem/search_service.py`
- Modify: `src/tracemem/retrieval.py`
- Modify: `src/tracemem/model_clients.py`
- Modify: `tests/test_observability.py`
- Modify: `tests/test_embedding_retry.py`

**Interfaces:**
- Produces: one 12-character hexadecimal `search_id` per Search operation.
- Extends: `HybridRetriever.retrieve(..., search_id: str | None = None)`.
- Produces: retrieval stage events correlated by `search_id`.
- Converts existing retry warnings to `event=embedding_retry`.

- [ ] **Step 1: Write failing successful-Search correlation/privacy test**

Use a fake retriever that captures its received `search_id` and returns a
literal `Candidate`. Assert `search_started`, `search_retrieved`,
`search_reranked`, and `search_completed` all contain the same hexadecimal ID,
while query/options/user markers never occur in logs.

```python
ids = re.findall(r"search_id=([0-9a-f]{12})", caplog.text)
assert len(ids) >= 4
assert len(set(ids)) == 1
assert retriever.search_id == ids[0]
for secret in ("SECRET_QUERY", "SECRET_OPTION", "SECRET_USER"):
    assert secret not in caplog.text
```

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/test_observability.py::test_search_logs_correlated_stages_without_content -q`

Expected: FAIL because Search emits no internal events and does not pass an ID.

- [ ] **Step 3: Implement Search and retrieval events**

Generate `uuid4().hex[:12]` at Search entry. Log counts/durations around route
planning, retrieval, reranking, packing, completion, degradation, and failure.
Pass `search_id` to `HybridRetriever`; log query embedding and per-channel
counts without logging query text or user ID.

```python
search_id = uuid4().hex[:12]
started = perf_counter()
log_event(
    logger,
    logging.INFO,
    "search_started",
    search_id=search_id,
    top_k=request.top_k,
    options=len(request.options or ()),
)
candidates = await self.retriever.retrieve(
    user_id=request.user_id,
    query=request.query,
    options=request.options,
    plan=plan,
    search_id=search_id,
)
```

- [ ] **Step 4: Write and implement rerank-degradation privacy test**

Use a reranker that raises `ModelTransportError("SECRET_RERANK_BODY")`. Require
`search_degraded stage=rerank error=ModelTransportError`, assert the response
still uses original candidates, and assert the exception message is absent.

- [ ] **Step 5: Convert embedding retry warnings and protect them with tests**

Update existing retry tests to capture `tracemem.model_clients` logs and assert:

```text
event=embedding_retry attempt=1 status=503 delay=0.5
```

For transport errors assert `error=ConnectError`; ensure authorization values,
input text, response bodies, and endpoint URLs are absent.

- [ ] **Step 6: Run full verification**

Run: `pytest -q`

Expected: all tests PASS.

Run: `python -m compileall -q src tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: exit code 0.

- [ ] **Step 7: Commit implementation**

```bash
git add src/tracemem/observability.py src/tracemem/config.py \
  src/tracemem/app.py src/tracemem/add_service.py \
  src/tracemem/search_service.py src/tracemem/retrieval.py \
  src/tracemem/model_clients.py tests/test_observability.py \
  tests/test_embedding_retry.py .env.example README.md \
  docs/superpowers/plans/2026-07-31-internal-observability.md
git commit -m "feat: add internal observability logs"
```
