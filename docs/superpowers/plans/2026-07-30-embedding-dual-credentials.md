# Embedding Dual-Credential Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send an optional `X-Embedding-Key` header alongside the existing bearer credential for OpenAI-compatible embedding requests.

**Architecture:** Add one optional setting, pass it through the existing model-bundle factory, and let `OpenAIEmbedder` construct embedding-only headers. Verify the complete settings-to-HTTP path with an in-process `httpx.MockTransport`.

**Tech Stack:** Python 3.10+, pydantic-settings, httpx, NumPy, pytest

## Global Constraints

- `TRACEMEM_EMBEDDING_API_KEY` remains the required bearer credential.
- `TRACEMEM_EMBEDDING_EXTRA_KEY` is optional and maps only to `X-Embedding-Key`.
- Empty or whitespace-only extra keys must omit `X-Embedding-Key`.
- Extraction and reranking requests must not receive the extra embedding credential.
- Credentials must not appear in logs or error messages.

---

### Task 1: Optional Embedding Credential

**Files:**
- Create: `tests/test_embedding_auth.py`
- Modify: `src/tracemem/config.py`
- Modify: `src/tracemem/runtime.py`
- Modify: `src/tracemem/model_clients.py`
- Modify: `.env.example`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Settings`, `build_model_bundle(settings, client)`, and `Embedder.embed(texts)`.
- Produces: `Settings.embedding_extra_key: str` and `OpenAIEmbedder(extra_api_key: str = "")`.

- [x] **Step 1: Write the failing tests**

```python
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
```

- [x] **Step 2: Run the tests to verify RED**

Run: `python -m pytest tests/test_embedding_auth.py -q`

Expected: the dual-credential test fails because `embedding_extra_key` is
ignored and `X-Embedding-Key` is absent.

- [x] **Step 3: Implement the minimal settings-to-header path**

Add to `Settings`:

```python
embedding_extra_key: str = ""
```

Pass it from `build_model_bundle()`:

```python
extra_api_key=settings.embedding_extra_key,
```

Accept and store it in `OpenAIEmbedder`, then build request headers:

```python
headers = {"Authorization": f"Bearer {self.api_key}"}
if self.extra_api_key.strip():
    headers["X-Embedding-Key"] = self.extra_api_key.strip()
```

Use `headers` only for the embedding request.

- [x] **Step 4: Document the optional credential**

Add this variable to `.env.example` and the README embedding example:

```dotenv
TRACEMEM_EMBEDDING_EXTRA_KEY=
```

Explain that it produces `X-Embedding-Key` only for embedding requests.

- [x] **Step 5: Run focused and full verification**

Run:

```bash
python -m pytest tests/test_embedding_auth.py -q
python -m pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` exits successfully.

- [x] **Step 6: Commit**

```bash
git add tests/test_embedding_auth.py src/tracemem/config.py src/tracemem/runtime.py src/tracemem/model_clients.py .env.example README.md docs/superpowers/plans/2026-07-30-embedding-dual-credentials.md
git commit -m "feat: support embedding dual credentials"
```
