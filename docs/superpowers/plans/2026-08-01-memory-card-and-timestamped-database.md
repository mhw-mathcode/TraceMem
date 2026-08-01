# Memory Card Reliability and Timestamped Database Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve useful memory cards when an extractor returns minor schema deviations, prove cards participate in retrieval, and create one fresh timestamped SQLite database for every server startup.

**Architecture:** `OpenAIExtractor` will keep the top-level response strict while validating cards independently, normalizing safe enum variations and dropping only malformed cards. A pure path helper in `app.py` will derive one timestamped database path at lifespan startup; that resolved path is then injected into the existing Add and Search services for the full process lifetime.

**Tech Stack:** Python 3.10+, FastAPI lifespan, Pydantic v2, HTTPX mock transport, SQLite/FTS5, pytest and pytest-asyncio.

## Global Constraints

- Do not change the public Add or Search API contract.
- Do not log message text, card text, model response bodies, user identifiers, or credentials.
- Do not invent semantic card content when the model returns no usable cards.
- Unknown optional `relation_hint` values normalize to `None`; invalid required card fields reject only that card.
- Timestamping is enabled by default for filesystem databases and leaves `:memory:` unchanged.
- Generate the timestamped path once per FastAPI lifespan and share it across all Add and Search requests in that lifespan.
- Keep old database files; never rename, overwrite, or automatically delete them.

---

### Task 1: Tolerant Per-Card Extraction and Retrieval Proof

**Files:**
- Create: `tests/test_card_extraction.py`
- Modify: `src/tracemem/model_clients.py:450-620`

**Interfaces:**
- Consumes: `OpenAIExtractor.extract(messages: Sequence[MessageForExtraction]) -> list[ExtractedCard]`, `AddService`, `HybridRetriever`, and the existing `Database` card tables.
- Produces: the same public extractor signature, with strict top-level validation, independent `_CardPayload` validation, safe enum normalization, and `card_extraction_parsed` aggregate logging.

- [ ] **Step 1: Write failing extractor contract tests**

Create `tests/test_card_extraction.py` with a small HTTPX responder that returns a complete OpenAI chat-completion envelope. Add tests that exercise the real `OpenAIExtractor` and assert these literal outcomes:

```python
@pytest.mark.asyncio
async def test_unknown_optional_relation_keeps_card(caplog):
    extractor = extractor_returning(
        {
            "cards": [{
                "kind": " Preference ",
                "subject": "user",
                "predicate": "dislikes",
                "object": "cilantro",
                "event_time": None,
                "polarity": " Negative ",
                "confidence": 0.9,
                "importance": 0.8,
                "source_message_indexes": [0],
                "relation_hint": "self-reported preference",
            }]
        }
    )

    cards = await extractor.extract([source_message("I dislike cilantro")])

    assert len(cards) == 1
    assert cards[0].kind == "preference"
    assert cards[0].polarity == "negative"
    assert cards[0].relation_hint is None
    assert "returned=1 accepted=1 normalized=1 rejected=0" in caplog.text
```

Add a second test with three returned cards: one valid card, one unsupported required `kind`, and one unknown source index. Assert that only the valid card is returned and the log reports `returned=3 accepted=1 normalized=0 rejected=2`. Ensure secret card/message content is absent from captured logs.

Add a third boundary test that captures the outgoing HTTP request and asserts the user prompt tells the upstream model the accepted `polarity` and `relation_hint` values. This protects the actual request contract rather than source text.

- [ ] **Step 2: Write a failing Add/Search integration test**

In the same file, use a real temporary `Database`, real `AddService`, real `OpenAIExtractor` backed by `httpx.MockTransport`, a deterministic local embedder, and real `HybridRetriever`. Return a card with `relation_hint="self-reported preference"`, call Add, and then retrieve with cards enabled.

Assert:

```python
assert response.success is True
assert database.count_cards("user-1") == 1
assert any(
    item.source_type == "card" and "cilantro" in item.content
    for item in candidates
)
assert "cards=1 raw_only=false" in caplog.text
```

The production change that makes this test pass is tolerant relation parsing; without it, current Add degrades to raw-only storage and the card assertion fails.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```powershell
py -3.11 -m pytest tests/test_card_extraction.py -q --basetemp .pytest_tmp_cards_red
```

Expected: failures show `ModelResponseError` for the observed unsupported relation, valid sibling cards are lost, and the prompt omits the complete enum contract.

- [ ] **Step 4: Implement strict-envelope, tolerant-card parsing**

In `src/tracemem/model_clients.py`:

- Replace whole-list `_ExtractionPayload` validation with an envelope model whose `cards` field is `list[Any]` while retaining `extra="forbid"` at the top level.
- Parse each raw card with `_CardPayload.model_validate(raw_card)` inside the loop and catch `ValidationError` per card.
- Add a private enum normalizer equivalent to:

```python
def _normalized_enum(value: str) -> str:
    return value.strip().casefold()
```

- Normalize `memory` to `fact`, whitespace, and case.
- Reject only the current card for unsupported required kind/polarity or unknown source indexes.
- Convert an unsupported optional relation to `None`.
- Count a card as normalized when at least one accepted enum value changes or an unsupported relation is removed.
- Emit exactly one safe aggregate event per valid top-level response:

```python
log_event(
    logger,
    logging.INFO,
    "card_extraction_parsed",
    returned=len(payload.cards),
    accepted=len(extracted),
    normalized=normalized_cards,
    rejected=rejected_cards,
)
```

- Expand the prompt to enumerate all accepted polarity and relation values and require `null` for no relation.

Do not include rejected values or card content in the log.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
py -3.11 -m pytest tests/test_card_extraction.py -q --basetemp .pytest_tmp_cards_green
```

Expected: all card extraction and integration tests pass.

- [ ] **Step 6: Commit the card reliability slice**

```powershell
git add tests/test_card_extraction.py src/tracemem/model_clients.py
git commit -m "fix: preserve valid memory cards"
```

### Task 2: One Timestamped Database Per Server Startup

**Files:**
- Create: `tests/test_timestamped_database.py`
- Modify: `src/tracemem/app.py:1-95`
- Modify: `tests/test_observability.py:180-215`

**Interfaces:**
- Consumes: `Settings.database_path`, FastAPI lifespan, and `Database(path)`.
- Produces: `_timestamped_database_path(path: Path, *, now: Callable[[], datetime] = _utc_now, process_id: int | None = None) -> Path`; the lifespan calls it once and reuses its result.

- [ ] **Step 1: Write failing path derivation tests**

Create `tests/test_timestamped_database.py` with literal expectations using a fixed UTC instant:

```python
def test_timestamped_database_path_inserts_utc_timestamp(tmp_path):
    result = _timestamped_database_path(
        tmp_path / "tracemem.db",
        now=lambda: datetime(2026, 8, 1, 6, 30, 52, 123456, tzinfo=timezone.utc),
    )
    assert result == tmp_path / "tracemem-20260801T063052123456Z.db"


def test_timestamped_database_path_preserves_memory_database():
    assert _timestamped_database_path(Path(":memory:")) == Path(":memory:")
```

Also cover a filename without an extension and a pre-existing primary candidate. For the collision case, pass `process_id=4321` and assert the fallback filename ends in `-p4321.db` and does not equal the existing path.

- [ ] **Step 2: Write a failing lifespan isolation test**

Create two app instances sequentially with the same base `tmp_path / "runtime.db"`, hash embeddings, disabled extraction, and disabled reranking. Enter each lifespan and capture:

```python
first_path = first_app.state.add_service.database.path
second_path = second_app.state.add_service.database.path
```

Assert both files exist, both names match
`runtime-\d{8}T\d{12}Z(?:-p\d+)?\.db`, the paths differ, and the configured base `runtime.db` does not exist. Within the first lifespan, assert the Add service and the retriever held by Search share the exact same `Database` object.

The production change that makes this test pass is resolving the timestamped path once in lifespan rather than constructing `Database(active_settings.database_path)` directly.

- [ ] **Step 3: Update the existing observability test expectation**

Change `test_lifespan_logs_database_initialization` to find the single created
`runtime-*.db`, assert its size is nonzero, and assert the timestamped filename
appears in the startup log. Keep the `:memory:` logging test unchanged.

- [ ] **Step 4: Run timestamp tests and verify RED**

Run:

```powershell
py -3.11 -m pytest tests/test_timestamped_database.py tests/test_observability.py::test_lifespan_logs_database_initialization -q --basetemp .pytest_tmp_db_red
```

Expected: import or filename assertions fail because no timestamp helper exists and startup still writes the base path.

- [ ] **Step 5: Implement timestamp path resolution**

In `src/tracemem/app.py`:

- Import `os`, `datetime`, `timezone`, and `Callable`.
- Add `_utc_now() -> datetime`.
- Add `_timestamped_database_path(...)` with the signature above.
- Return `Path(":memory:")` unchanged.
- Format UTC time as `%Y%m%dT%H%M%S%fZ`.
- Insert the timestamp before the final suffix, or append it for suffixless names.
- If the primary candidate exists, append `-p<process id>` before the suffix; if that path also exists, append an increasing numeric suffix until an unused path is found.
- In lifespan, call the helper once and construct `Database` from that resolved path.

Do not create or remove files inside the helper; schema initialization remains owned by `Database.initialize()`.

- [ ] **Step 6: Run timestamp and observability tests and verify GREEN**

Run:

```powershell
py -3.11 -m pytest tests/test_timestamped_database.py tests/test_observability.py -q --basetemp .pytest_tmp_db_green
```

Expected: all selected tests pass and startup logs the generated database path.

- [ ] **Step 7: Commit the database lifecycle slice**

```powershell
git add tests/test_timestamped_database.py tests/test_observability.py src/tracemem/app.py
git commit -m "feat: create a database per server startup"
```

### Task 3: Deployment Documentation and Full Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the implemented default database naming behavior.
- Produces: operator documentation explaining where the active database path appears and that restarts retain old databases.

- [ ] **Step 1: Update operator documentation**

In the README configuration/startup section, document that
`TRACEMEM_DATABASE_PATH=data/tracemem.db` is a base filename and that startup
creates `data/tracemem-<UTC timestamp>.db`. State that Add/Search in the running
process share the file, restarting creates a new file, old files remain, and
the exact active path is shown by `event=database_initialized`.

- [ ] **Step 2: Run formatting and the complete test suite**

Run:

```powershell
git diff --check
py -3.11 -m pytest -q --basetemp .pytest_tmp_full
```

Expected: no whitespace errors and the entire suite passes.

- [ ] **Step 3: Inspect the final diff for scope and secrets**

Run:

```powershell
git diff --stat HEAD~2
git diff HEAD~2 -- src/tracemem/model_clients.py src/tracemem/app.py README.md
rg -n "sk-|Bearer |X-Embedding-Key" tests src README.md
```

Confirm the diff contains no credential values, request content logging, unrelated refactors, database deletion, or public API changes.

- [ ] **Step 4: Commit documentation**

```powershell
git add README.md docs/superpowers/plans/2026-08-01-memory-card-and-timestamped-database.md
git commit -m "docs: explain timestamped database lifecycle"
```

- [ ] **Step 5: Request code review and resolve findings**

Use `superpowers:requesting-code-review` against the complete implementation diff. Apply only verified findings through `superpowers:receiving-code-review`, rerun the focused test for each correction, and rerun the complete suite before completion.
