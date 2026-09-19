# Multimodal Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept ordered text/image Add content, retrieve image-only memories by visible content, and return original image parts within competition limits.

**Architecture:** Validate and retain original parts separately from an image-derived text index. A separately configured `gpt-4o-mini` Chat Completions vision client describes each image before the existing embedding/FTS/card pipeline; Search loads original parts only for selected episodes and budgets image bytes before response construction.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic 2, SQLite/FTS5, httpx, Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-multimodal-memory-design.md`

## Global Constraints

- Add content is a nonempty string or nonempty ordered array of `text` and `image_url` parts; Search query stays text-only.
- Image Data URIs are inline Base64 JPEG, PNG, or WebP only; one decoded image is at most 10 MiB, one Add at most 30 MiB, and one Search response at most 30 MiB.
- The original part sequence and Data URIs survive persistence and Search unchanged; text-only behavior remains compatible.
- A vision failure must fail image-bearing Add with 502 and permit retry; never silently accept an unsearchable image.
- Credentials, Data URIs, and vision response bodies must not be logged.
- `docs/` and `tests/` are ignored by `.gitignore`; use `git add -f` only for specific new plan/test files, never force-add broad directories.

## Review Focus

- Invalid Base64 with a plausible prefix must return 422, not crash or contact a model (Task 1 test).
- Multiple individually valid images exceeding 30 MiB across messages must return 422 before a database claim (Task 1 test).
- A valid image with no text must still be indexed and found by a text query (Task 4 test).
- An image result that does not fit Search's remaining budget must be skipped whole while later smaller results remain eligible (Task 5 test).
- Retrying the same request ID after a failed vision call must succeed without duplicate episodes (Task 4 test).

## File map

- `src/tracemem/multimodal.py`: ordered-part types, Data URI validation, byte counting, and indexing-text assembly.
- `src/tracemem/vision.py`: one OpenAI-compatible vision client and a protocol for deterministic tests.
- `src/tracemem/api_models.py`: public Add/Search content unions and aggregate Add validation.
- `src/tracemem/config.py`, `runtime.py`, `app.py`: independent vision settings, wiring, and error mapping.
- `src/tracemem/domain.py`, `db.py`: separate original-payload storage and selected-episode loading.
- `src/tracemem/add_service.py`, `search_service.py`: vision indexing and ordered/budgeted response paths.
- `pyproject.toml`, `README.md`, `.env.example`: Pillow dependency and operator setup.
- `tests/test_multimodal_*.py`: focused protocol, client, storage, and end-to-end regressions.

### Task 1: Public payload contract and bounded image validation

**Files:** Create `src/tracemem/multimodal.py`, `tests/test_multimodal_api.py`; modify `src/tracemem/api_models.py`, `pyproject.toml`.

**Interfaces:** Produce `TextPart`, `ImagePart`, `Content = str | list[TextPart | ImagePart]`, `image_bytes(content) -> int`, `context_text(content) -> str`, and `index_text(content, descriptions) -> str`; `AddRequest` validates aggregate bytes. Later tasks import these names.

- [ ] **Step 1: Write failing tests.** Test a valid mixed array's order, an image-only array, legacy string, empty text, remote URL, invalid Base64, JPEG/PNG/WebP content-type mismatch, >10 MiB image, and >30 MiB aggregate. Use a tiny valid PNG fixture and test overridable limit constants rather than allocating 30 MiB in every test. Example assertion:

```python
def test_rejects_remote_image_url():
    payload = add_payload([{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}])
    with pytest.raises(ValidationError):
        AddRequest.model_validate(payload)
```

- [ ] **Step 2: Verify red.** Run `python -m pytest tests/test_multimodal_api.py -q`; expect validation failures for the new accepted array and/or imports of missing helpers.
- [ ] **Step 3: Implement minimal validation.** Use discriminated Pydantic models (`type: Literal[...]`, `extra="forbid"`), strict Base64 decoding, a pre-decode encoded-length guard, decoded byte limits, and Pillow `Image.open(...).verify()` plus `format`/MIME matching. Implement the aggregate limit in an `AddRequest` model validator. Keep output Data URI verbatim. Expose a pure indexing-text helper that never includes Base64.

```python
ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
Content = Annotated[str | list[ContentPart], Field(union_mode="left_to_right")]
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_IMAGE_BYTES = 30 * 1024 * 1024
MAX_RESPONSE_IMAGE_BYTES = 30 * 1024 * 1024
```

- [ ] **Step 4: Verify green and schema.** Run `python -m pytest tests/test_multimodal_api.py -q` and inspect `AddRequest.model_json_schema()` for both string and ordered-array shapes.
- [ ] **Step 5: Commit.** `git add src/tracemem/multimodal.py src/tracemem/api_models.py pyproject.toml`; `git add -f tests/test_multimodal_api.py`; `git commit -m "feat: validate multimodal memory payloads"`.

### Task 2: Vision description client and independent configuration

**Files:** Create `src/tracemem/vision.py`, `tests/test_multimodal_vision.py`; modify `src/tracemem/config.py`, `src/tracemem/runtime.py`, `src/tracemem/app.py`, `.env.example`.

**Interfaces:** Produce `VisionDescriber.describe(image_url: str, context: str) -> str` async protocol; `OpenAIVisionDescriber` calls Chat Completions using `image_url` part; `ModelBundle.vision` exposes it. A missing key raises a distinct `VisionUnavailable` at image use time.

- [ ] **Step 1: Write failing tests.** Use `httpx.MockTransport` to inspect `model="gpt-4o-mini"`, ordered text/image parts, unchanged Data URI, auth header, and parsed nonempty description. Test HTTP failure, invalid/empty model content, and missing key. Assert request/exception strings do not contain Base64.

```python
@pytest.mark.asyncio
async def test_vision_uses_inline_image_url():
    result = await describer.describe(PNG_DATA_URI, "user caption")
    assert result == "A red bicycle beside a tree."
    assert captured["messages"][1]["content"][1]["image_url"]["url"] == PNG_DATA_URI
```

- [ ] **Step 2: Verify red.** Run `python -m pytest tests/test_multimodal_vision.py -q`; expect missing client/protocol failures.
- [ ] **Step 3: Implement client and settings.** Add `vision_url`, `vision_api_key`, `vision_model="gpt-4o-mini"`; no image credentials check at startup. Reuse the existing shared `httpx.AsyncClient` and endpoint URL normalization. Limit returned description length and reject blank output. Map `VisionUnavailable`/vision transport/schema errors to a 502 Add response without exposing image content.

```python
payload = {"model": self.model, "messages": [
    {"role": "system", "content": VISION_PROMPT},
    {"role": "user", "content": [
        {"type": "text", "text": context or "Describe this image for search."},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]},
]}
```

- [ ] **Step 4: Verify green.** Run `python -m pytest tests/test_multimodal_vision.py -q` and an import smoke check for `create_app`.
- [ ] **Step 5: Commit.** `git add src/tracemem/vision.py src/tracemem/config.py src/tracemem/runtime.py src/tracemem/app.py .env.example`; `git add -f tests/test_multimodal_vision.py`; `git commit -m "feat: configure vision descriptions"`.

### Task 3: Persist original parts apart from search text

**Files:** Modify `src/tracemem/domain.py`, `src/tracemem/db.py`; create `tests/test_multimodal_storage.py`.

**Interfaces:** `EpisodeDraft.original_content: Content | None = None`; `Database.load_original_contents(episode_ids: Sequence[str]) -> dict[str, list[ContentPart]]`. Episode `content` remains the index/search string. No Base64 enters candidate text or vector records.

- [ ] **Step 1: Write failing tests.** Claim and commit an episode with image parts, then assert its indexed text appears in FTS/vector candidate while `load_original_contents` returns the exact ordered parts. Check text-only episodes have no original-payload row. Reinitialize an existing DB and confirm old records remain readable.

```python
assert database.load_original_contents([episode.id])[episode.id] == original_parts
assert "base64," not in database.load_episode_vectors("u")[0].candidate.content
```

- [ ] **Step 2: Verify red.** Run `python -m pytest tests/test_multimodal_storage.py -q`; expect missing storage interface.
- [ ] **Step 3: Implement separate `episode_payloads` table.** Use `episode_id` primary/foreign key and `content_json TEXT NOT NULL`; write rows atomically inside `commit_add`, and query by selected IDs only. Serialize through Pydantic `model_dump(mode="json")` and parse back through the public content adapter. Keep existing episode schema and all text-only reads unchanged.

```sql
CREATE TABLE IF NOT EXISTS episode_payloads (
    episode_id TEXT PRIMARY KEY REFERENCES episodes(id),
    content_json TEXT NOT NULL
);
```

- [ ] **Step 4: Verify green.** Run `python -m pytest tests/test_multimodal_storage.py -q` plus existing database tests if present.
- [ ] **Step 5: Commit.** `git add src/tracemem/domain.py src/tracemem/db.py`; `git add -f tests/test_multimodal_storage.py`; `git commit -m "feat: retain original multimodal episodes"`.

### Task 4: Index visible image content during Add

**Files:** Modify `src/tracemem/add_service.py`, `src/tracemem/app.py`; create `tests/test_multimodal_add.py`.

**Interfaces:** `AddService(..., vision: VisionDescriber)`; construct one indexed string per message before embedding; pass original array to `EpisodeDraft.original_content`; text-only path keeps existing content. Use Task 2 `describe` and Task 1 `index_text`.

- [ ] **Step 1: Write failing tests.** With deterministic fake vision/embedding/extractor, Add a pure image, assert its index includes the visual description and a text retrieval query can find the episode. Assert neither embedder nor extractor receives Base64. Make vision fail once; assert `AddDependencyError`/failed claim, retry same request ID succeeds, and one episode exists. Assert a text-only Add never calls vision. The HTTP 502 mapping is tested in Task 5.

```python
await add_service.add(image_only_request)
assert "red bicycle" in database.load_episode_vectors("u")[0].candidate.content
assert all("base64," not in value for value in embedder.seen_texts)
```

- [ ] **Step 2: Verify red.** Run `python -m pytest tests/test_multimodal_add.py -q`; expect string-only indexing or missing vision dependency failure.
- [ ] **Step 3: Implement indexing before embedding.** Describe images after claim but before embedding; on vision failure call `mark_add_failed` and raise an Add dependency error with a generic message. Keep original message for hash and persistence; pass indexed strings into embeddings, lexical processing, and extraction. Ensure indexing strings for image-only messages are nonempty.

```python
indexed_texts = []
for message in request.messages:
    descriptions = []
    if isinstance(message.content, list):
        for part in message.content:
            if isinstance(part, ImagePart):
                descriptions.append(await self.vision.describe(part.image_url.url, context_text(message.content)))
    indexed_texts.append(index_text(message.content, descriptions))
vectors = await self.embedder.embed(indexed_texts)
```
- [ ] **Step 4: Verify green.** Run `python -m pytest tests/test_multimodal_add.py -q` and all Task 1–3 tests.
- [ ] **Step 5: Commit.** `git add src/tracemem/add_service.py src/tracemem/app.py`; `git add -f tests/test_multimodal_add.py`; `git commit -m "feat: index image descriptions"`.

### Task 5: Emit ordered multimodal Search results within 30 MiB

**Files:** Modify `src/tracemem/search_service.py`, `src/tracemem/api_models.py`; create `tests/test_multimodal_search.py`, `tests/test_multimodal_http.py`.

**Interfaces:** For episode IDs chosen by packer, use `Database.load_original_contents`; `SearchResult.content` accepts `Content`. Score normalization uses only emitted candidates.

- [ ] **Step 1: Write failing tests.** Search for an image-only memory and assert exact ordered array/Data URI. Search with two candidates of two 10 MiB images each and a later tiny candidate; assert the first large result and later tiny result survive while the second large one is skipped whole. Assert legacy text results remain strings and `top_k` is never exceeded. Add FastAPI HTTP tests with an in-memory database and fake vision client: POST image-only Add then text Search, assert status 200 and exact original JSON part sequence; assert 422 for bad image input and both content shapes in OpenAPI.

```python
response = await search_service.search(query)
assert response.data[0].content == original_parts
assert sum(image_bytes(item.content) for item in response.data) <= 30 * 1024 * 1024
```

- [ ] **Step 2: Verify red.** Run `python -m pytest tests/test_multimodal_search.py tests/test_multimodal_http.py -q`; expect string output or an oversized image aggregate.
- [ ] **Step 3: Implement result budgeting.** Ask packer for enough ranked candidates to replace skipped oversized results; iterate candidates in rank order, load original parts for episode IDs, skip whole oversized results, stop at the requested/result limits, and normalize scores over emitted candidates. If no images are present, preserve existing rendering.

```python
payloads = self.retriever.database.load_original_contents(
    [item.id for item in ranked if item.source_type == "episode"]
)
emitted = []
used_bytes = 0
for candidate in ranked:
    content = payloads.get(candidate.id, render_candidate(candidate))
    size = image_bytes(content)
    if used_bytes + size > MAX_RESPONSE_IMAGE_BYTES:
        continue
    emitted.append((candidate, content))
    used_bytes += size
    if len(emitted) == limit:
        break
```
- [ ] **Step 4: Verify green.** Run `python -m pytest tests/test_multimodal_search.py tests/test_multimodal_http.py -q` and all multimodal tests.
- [ ] **Step 5: Commit.** `git add src/tracemem/search_service.py src/tracemem/api_models.py`; `git add -f tests/test_multimodal_search.py tests/test_multimodal_http.py`; `git commit -m "feat: return budgeted multimodal search evidence"`.

### Task 6: Documentation and end-to-end verification

**Files:** Modify `README.md`.

**Interfaces:** HTTP `/add` and `/search` preserve their existing route, authentication, and envelope schemas; only `content` gains the union.

- [ ] **Step 1: Document setup and behavior.** Add a mixed-part Add JSON example and `TRACEMEM_VISION_URL`, `TRACEMEM_VISION_API_KEY`, and `TRACEMEM_VISION_MODEL=gpt-4o-mini` setup notes; document 10/30 MiB limits, 502 behavior, text-only compatibility, and the vision-description retrieval trade-off. Do not include a real key.

```json
{"role":"user","content":[{"type":"text","text":"A trip photo"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<image bytes encoded in base64>"}}]}
```
- [ ] **Step 2: Verify documentation against implementation.** Check each new setting and example field against `config.py` and `api_models.py`; run `python -m pytest -q`, `python -m compileall -q src`, and `git diff --check`. If no live OpenAI key is available, explicitly report that real-provider smoke testing was not performed; do not use a user's key without consent.
- [ ] **Step 3: Commit.** `git add README.md`; `git commit -m "docs: explain multimodal memory setup"`.

## Final review

- [ ] Inspect the complete branch diff against this plan and spec; verify there are no Base64 values in logs, text embedding inputs, or error messages.
- [ ] Confirm every review-focus case has an automated test and every test was observed red before implementation.
- [ ] Run the full test command afresh before any completion claim or PR/merge action.
