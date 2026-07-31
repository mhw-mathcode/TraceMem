# TraceMem

TraceMem is a temporal evidence-ledger memory service for the Agent Memory Challenge.

## HTTP API

TraceMem supports Python 3.10 and newer. Install the project, then start the
FastAPI service from the repository root:

```bash
python -m pip install .
python -m uvicorn main:app --host 0.0.0.0 --port 8888
```

When running directly from a source checkout without installing the package,
add `src` to `PYTHONPATH`:

```bash
PYTHONPATH=src python -m uvicorn main:app --host 0.0.0.0 --port 8888
```

The competition endpoints are:

```text
POST /add
POST /search
```

Interactive API documentation is available at `/docs`.

## API authentication

Both competition endpoints require the same `X-API-Key` request header.
Generate a strong key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Put the generated value in the server's `.env` file:

```dotenv
TRACEMEM_API_KEY=replace-with-your-generated-secret
```

Use that same value in the competition platform's memory-system Key field.
Do not commit or share the real `.env` file. TraceMem refuses to start when
`TRACEMEM_API_KEY` is empty.

Example requests:

```bash
curl -X POST http://127.0.0.1:8888/add \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace-with-your-generated-secret" \
  -d '{
    "request_id": "request-1",
    "user_id": "user-1",
    "session_id": "session-1",
    "messages": [
      {"role": "user", "content": "Alice likes jasmine tea."}
    ]
  }'

curl -X POST http://127.0.0.1:8888/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: replace-with-your-generated-secret" \
  -d '{
    "user_id": "user-1",
    "query": "What does Alice like?",
    "top_k": 5
  }'
```

After changing `.env`, restart the Uvicorn process so it loads the new key.
Swagger at `/docs` remains public; click **Authorize** and enter the key to
test `/add` and `/search`.

## Model configuration

Copy the values in `.env.example` to `.env`. A local `.env` is already
provided and ignored by Git. Embedding and LLM extraction use independent
OpenAI-compatible endpoints and credentials:

```dotenv
TRACEMEM_EMBEDDING_URL=https://your-provider/v1/embeddings
TRACEMEM_EMBEDDING_API_KEY=your-embedding-key
TRACEMEM_EMBEDDING_EXTRA_KEY=your-optional-x-embedding-key
TRACEMEM_EMBEDDING_MODEL=your-embedding-model
TRACEMEM_EMBEDDING_MAX_CONCURRENCY=3

TRACEMEM_LLM_URL=https://your-provider/v1/chat/completions
TRACEMEM_LLM_API_KEY=your-llm-key
TRACEMEM_LLM_MODEL=your-llm-model

TRACEMEM_RERANK_MODE=disabled
```

The URLs may be complete request endpoints or OpenAI-compatible base URLs
ending in `/v1`; TraceMem appends `/embeddings` or `/chat/completions` only
for base URLs. Embedding and LLM can use different providers. Rerank remains
available in code but is disabled by default. When
`TRACEMEM_EMBEDDING_EXTRA_KEY` is non-empty, embedding requests also include
it as the `X-Embedding-Key` header; other model requests never receive it.
Remote embedding calls retry connection failures and HTTP 429/500/502/503/504
responses until they succeed, using exponential backoff and `Retry-After` when
provided. `TRACEMEM_EMBEDDING_MAX_CONCURRENCY` limits simultaneous embedding
HTTP attempts and defaults to `3`. `TRACEMEM_MODEL_MAX_RETRIES` continues to
limit extraction and reranking retries; it does not limit embedding retries.

## LoCoMo retrieval evaluation

Run the deterministic offline baseline:

```powershell
$env:PYTHONPATH='src'
python scripts/run_locomo_retrieval.py data/locomo10.json --model-mode offline
```

After filling in both API keys, run a small remote-model smoke test:

```powershell
$env:PYTHONPATH='src'
python scripts/run_locomo_retrieval.py data/locomo10.json `
  --model-mode bailian --samples 1 --max-questions 20
```

The result JSON and its paired SQLite database are written under `outputs/`.
Use a fresh output name when changing embedding models so vectors with
different dimensions are never mixed.
