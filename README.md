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

For multimodal memory, a message's `content` may instead be an ordered array
of nonempty text parts and inline images. A message may contain only images:

```json
{
  "request_id": "request-photo-1",
  "user_id": "user-1",
  "session_id": "session-1",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Photo from the trip"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64 image bytes>"}}
    ]
  }]
}
```

Use an actual Base64 Data URI in place of the bracketed example value; remote
image URLs are not accepted. JPEG, PNG, and WebP are supported. Each decoded
image must be at most 10 MiB; all images in one Add request must total at most
30 MiB. Search returns the original parts in order for image-bearing episodes,
with at most 30 MiB of decoded images across one response. Results that would
exceed that response limit are omitted whole. Plain strings continue to work
for text-only messages and results.

Configure a vision-capable OpenAI-compatible Chat Completions endpoint to make
images searchable by what they depict:

```dotenv
TRACEMEM_VISION_URL=https://api.openai.com/v1/chat/completions
TRACEMEM_VISION_API_KEY=replace-with-your-vision-api-key
TRACEMEM_VISION_MODEL=gpt-4o-mini
```

TraceMem asks the vision model for a factual description at Add time and
indexes that description with its existing text search. The original image is
retained separately and returned on Search. This supports text-to-image
retrieval, including image-only messages, but not image-to-image queries.
Without a vision key, text-only requests still work; an image-bearing Add
returns HTTP 502 rather than accepting an image it cannot index. Image
description quality affects retrieval quality. Vision API use may incur cost.

After changing `.env`, restart the Uvicorn process so it loads the new key.
Swagger at `/docs` remains public; click **Authorize** and enter the key to
test `/add` and `/search`.

## Internal logs

TraceMem writes grep-friendly internal events to standard error. With the
documented `nohup` command they are collected in `/tmp/tracemem.log` alongside
Uvicorn access logs. Set `TRACEMEM_LOG_LEVEL=INFO` (the default) to record
startup, database initialization, Add/Search stages, durations, degradation,
and embedding retries. Logs contain identifiers and counts but never message
content, query text, user/session IDs, credentials, or model bodies.

## Database lifecycle

`TRACEMEM_DATABASE_PATH` selects the base SQLite filename and defaults to
`data/tracemem.db`. Each service startup creates a fresh database by inserting
a UTC timestamp before the extension, for example
`data/tracemem-20260801T143052123456Z.db`. All Add and Search requests handled
by that running process share the generated file. Restarting the service
creates another database; earlier files are retained and are never renamed or
deleted automatically. The active absolute path is recorded by the
`event=database_initialized` startup log.

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
provided. Other HTTP errors and malformed successful responses are attempted
ten times; rejected multi-item batches are then isolated once per item. Inputs
that still fail are stored with an empty vector and remain available to
lexical/BM25 retrieval, while Add and Search continue with their normal
successful response schemas. `TRACEMEM_EMBEDDING_MAX_CONCURRENCY` limits
simultaneous embedding HTTP attempts and defaults to `3`.
`TRACEMEM_MODEL_MAX_RETRIES` continues to limit extraction and reranking
retries; it does not limit embedding retries.

Memory extraction explicitly constrains card kind, polarity, and relation
values. Minor formatting differences and unknown optional relation hints are
normalized without discarding an otherwise valid card, and one malformed card
does not discard valid cards returned beside it. The
`event=card_extraction_parsed` log reports returned, accepted, normalized, and
rejected counts without exposing card or message content.

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
