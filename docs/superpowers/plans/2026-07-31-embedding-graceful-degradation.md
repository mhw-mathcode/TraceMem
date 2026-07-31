# Embedding Graceful Degradation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retry non-temporary embedding failures ten times, isolate rejected batch items, and return lexical-only results instead of propagating model-related HTTP 502 responses.

**Architecture:** Keep the policy inside `OpenAIEmbedder` so Add, card generation, and Search all receive exactly one vector per input. Temporary failures retain persistent retry; bounded failures return zero-dimensional vectors after batch recovery and per-item isolation. Existing storage and retrieval code accepts empty vectors, while observability gains safe counts and payload-size metadata.

**Tech Stack:** Python 3.10+, asyncio, httpx, NumPy, FastAPI service layer, SQLite FTS5, pytest.

## Global Constraints

- Non-temporary embedding failures receive ten batch attempts with a fixed 0.5-second delay between attempts.
- HTTP 429/500/502/503/504, transport failures, and premature streams continue retrying indefinitely with the existing backoff.
- A failed multi-item batch gets one per-item isolation request; failed items become zero-dimensional vectors.
- Add and Search public schemas remain unchanged and model degradation must not return HTTP 502.
- Empty vectors remain lexical-only and must never produce semantic similarity scores.
- Logs may contain statuses, counts, attempts, delays, and character lengths, but no content, response body, URL, identity, or credential.

---

### Task 1: Bounded embedding recovery and isolation

**Files:**
- Modify: `src/tracemem/model_clients.py:201-315`
- Modify: `tests/test_embedding_retry.py:1-285`

**Interfaces:**
- Consumes: `OpenAIEmbedder.embed(texts: Sequence[str]) -> list[np.ndarray]`, the existing HTTP client, semaphore, retry delay helper, and safe `log_event` helper.
- Produces: an internal response parser and request loop that always returns one real or empty `np.ndarray` per input after bounded failures; no public signature changes.

- [ ] **Step 1: Replace the permanent-error expectation with a failing ten-attempt degradation test**

```python
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        embedder = OpenAIEmbedder(
            client=client,
            url="https://SECRET_ENDPOINT.test/v1/embeddings",
            api_key="SECRET_API_KEY",
            model="embedding",
            sleep=record_sleep,
        )
        with caplog.at_level(logging.WARNING, logger="tracemem.model_clients"):
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
    assert "SECRET" not in caplog.text
```

- [ ] **Step 2: Add failing tests for multi-item isolation and malformed HTTP 200 responses**

```python
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
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
```

- [ ] **Step 3: Run focused tests and verify RED**

Run: `python -m pytest tests/test_embedding_retry.py -q --basetemp .pytest_tmp_embedding_red`

Expected: the old implementation raises `ModelTransportError` for 401 and `ModelResponseError` for invalid successful responses instead of returning empty vectors.

- [ ] **Step 4: Implement safe metrics, response parsing, bounded retries, and isolation**

Add constants and focused internal methods in `OpenAIEmbedder`:

```python
_BOUNDED_EMBEDDING_ATTEMPTS = 10
_BOUNDED_EMBEDDING_DELAY_SECONDS = 0.5


def _embedding_input_metrics(texts: Sequence[str]) -> dict[str, int]:
    lengths = [len(text) for text in texts]
    return {
        "batch_items": len(lengths),
        "max_chars": max(lengths, default=0),
        "total_chars": sum(lengths),
    }
```

Move response validation into a static
`_parse_vectors(response: httpx.Response, expected_count: int) -> list[np.ndarray]`
method. It sorts by `index`, requires exactly `expected_count` entries, converts
each vector to one-dimensional `float32`, rejects zero length and inconsistent
dimensions, and raises `ModelResponseError("Embedding response has an invalid schema")`
for `KeyError`, `TypeError`, `ValueError`, JSON decode errors, or failed checks.

Implement
`_embed_batch(headers: dict[str, str], batch: Sequence[str], bounded_attempts: int) -> list[np.ndarray] | None`
with these exact branches:

```python
if response.status_code in _RETRYABLE_STATUS_CODES:
    # Existing persistent retry/backoff path.
elif response.is_error:
    # Increment bounded attempt, log status plus metrics, return None at limit.
else:
    # Parse vectors; schema errors use the same bounded count and fixed delay.
```

The retry loop keeps separate `temporary_attempt` and `bounded_attempt`
counters. A temporary failure increments only `temporary_attempt`; a successful
HTTP exchange or bounded failure resets `temporary_attempt` to zero. A bounded
failure increments only `bounded_attempt`, returns `None` when it equals
`bounded_attempts`, and otherwise sleeps exactly 0.5 seconds. Cancellation is
not caught.

In `embed`, call `_embed_batch(..., bounded_attempts=10)`. When it returns `None` for multiple inputs, emit `embedding_batch_isolation` and call `_embed_batch(..., bounded_attempts=1)` once per input. Substitute `np.asarray([], dtype=np.float32)` for each failed item, preserve order, and emit `embedding_degraded` with `real_vectors` and `empty_vectors`. Normalize any cross-batch dimension mismatch to an empty vector instead of raising a model error.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `python -m pytest tests/test_embedding_retry.py -q --basetemp .pytest_tmp_embedding_green`

Expected: all embedding retry, isolation, concurrency, cancellation, and secret-redaction tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add -- src/tracemem/model_clients.py tests/test_embedding_retry.py
git commit -m "feat: degrade rejected embedding inputs"
```

---

### Task 2: Verify lexical-only Add and Search behavior

**Files:**
- Modify: `src/tracemem/add_service.py:158-255`
- Modify: `src/tracemem/retrieval.py:122-220`
- Modify: `tests/test_observability.py:1-430`

**Interfaces:**
- Consumes: `OpenAIEmbedder.embed` returning zero-dimensional vectors, `Database.commit_add`, `HybridRetriever.retrieve`, and `cosine_candidates` zero-vector filtering.
- Produces: `empty_vectors` fields on existing completion events and an integration test proving empty vectors are stored and retrieved lexically without exceptions.

- [ ] **Step 1: Add a failing lexical-only integration test**

```python
class EmptyEmbedder:
    async def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray([], dtype=np.float32) for _text in texts]


@pytest.mark.asyncio
async def test_empty_embeddings_commit_and_search_lexically(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = Database(tmp_path / "lexical-only.db")
    database.initialize()
    add_service = AddService(
        database=database,
        embedder=EmptyEmbedder(),
        extractor=DisabledExtractor(),
    )
    response = await add_service.add(
        AddRequest(
            request_id="lexical-only",
            user_id="user",
            session_id="session",
            messages=[MemoryMessage(role="user", content="purple bicycles")],
        )
    )
    retriever = HybridRetriever(
        database=database,
        embedder=EmptyEmbedder(),
        cards_enabled=False,
        state_enabled=False,
    )
    candidates = await retriever.retrieve(
        user_id="user",
        query="purple bicycles",
        search_id="abcdef123456",
    )

    assert response.success is True
    assert [candidate.content for candidate in candidates] == ["purple bicycles"]
    assert "empty_vectors=1" in caplog.text
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_observability.py::test_empty_embeddings_commit_and_search_lexically -q --basetemp .pytest_tmp_lexical_red`

Expected: lexical retrieval already works, but the required `empty_vectors=1` diagnostics are missing.

- [ ] **Step 3: Add empty-vector counts to stage logs**

In `AddService.add`, add this field to episode and card completion events:

```python
empty_vectors=sum(1 for vector in vectors if vector.size == 0)
```

Use `card_vectors` for the card event. In `HybridRetriever.retrieve`, add the same count for `query_vectors` to `search_query_embedding_completed`. Do not log vector values or source text.

- [ ] **Step 4: Run observability tests and verify GREEN**

Run: `python -m pytest tests/test_observability.py -q --basetemp .pytest_tmp_observability_green`

Expected: all observability and lexical-only integration tests pass without secrets in captured logs.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- src/tracemem/add_service.py src/tracemem/retrieval.py tests/test_observability.py
git commit -m "feat: expose lexical embedding degradation"
```

---

### Task 3: Document and verify the operational behavior

**Files:**
- Modify: `README.md:108-118`

**Interfaces:**
- Consumes: the finalized retry and lexical fallback behavior from Tasks 1 and 2.
- Produces: an operator-facing explanation of which failures retry forever, which retry ten times, and what retrieval remains after degradation.

- [ ] **Step 1: Update the model-configuration documentation**

Replace the existing retry paragraph with wording that states:

```markdown
Remote embedding calls retry connection failures and HTTP 429/500/502/503/504
responses until they succeed. Other HTTP errors and malformed successful
responses are attempted ten times; rejected multi-item batches are then
isolated once per item. Inputs that still fail are stored with an empty vector
and remain available to lexical/BM25 retrieval, while Add and Search continue
with their normal successful response schemas.
```

Keep the existing concurrency and `TRACEMEM_MODEL_MAX_RETRIES` explanation.

- [ ] **Step 2: Run complete verification**

Run: `python -m pytest -q --basetemp .pytest_tmp_full`

Expected: all tests pass.

Run: `python -m compileall -q src tests`

Expected: exit code 0 with no output.

Run: `git diff --check`

Expected: exit code 0 with no whitespace errors.

- [ ] **Step 3: Commit Task 3**

```powershell
git add -- README.md
git commit -m "docs: explain embedding degradation"
```

- [ ] **Step 4: Request code review and address only verified findings**

Use `superpowers:requesting-code-review` against the implementation commits. If review reports issues, use `superpowers:receiving-code-review`, reproduce each issue, fix it test-first, and rerun complete verification.
