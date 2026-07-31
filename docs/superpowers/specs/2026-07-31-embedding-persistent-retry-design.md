# Embedding Persistent Retry Design

## Goal

Keep Add and Search requests alive while the embedding provider is temporarily
unavailable, returning to the evaluation platform only after embedding succeeds
or the calling task is cancelled.

## Scope

The change applies only to `OpenAIEmbedder`. Extraction and reranking retain
their existing bounded retry behavior. The public Add, Search, and health API
contracts do not change.

## Retry Classification

The embedder retries indefinitely for failures considered temporary:

- HTTP 429, 500, 502, 503, and 504 responses.
- Connection failures and request timeouts.
- Premature TLS stream termination.

Other HTTP errors, including 400, 401, and 403, fail immediately because another
identical request cannot correct an invalid request or credential.

## Backoff

Temporary failures use exponential backoff starting at 0.5 seconds and capped at
10 seconds. When a retryable response supplies a valid `Retry-After` delay, the
embedder honors it without allowing a negative delay.

There is no retry-count or elapsed-time limit. Task cancellation is not caught,
so cancellation propagated by the server stops the retry loop naturally. A raw
client disconnect is not assumed to cancel a non-streaming ASGI handler.

## Concurrency

One semaphore shared by each `OpenAIEmbedder` instance limits embedding HTTP
exchanges to three concurrent operations. A request holds a semaphore permit for
one HTTP attempt only and releases it during backoff, allowing other requests to
make progress while it waits.

## Data Flow

1. Add or Search requests an embedding.
2. The embedder waits for a concurrency permit.
3. It sends one HTTP request and releases the permit.
4. A successful response is validated and returned.
5. A temporary failure waits according to the backoff policy and returns to
   step 2.
6. A permanent HTTP failure or invalid successful response is returned as the
   existing model error.
7. Caller cancellation exits the loop and propagates without another attempt.

## Configuration

Persistent retry is the fixed behavior of the remote embedder for this
evaluation integration. The concurrency limit defaults to three and is
configurable through:

`TRACEMEM_EMBEDDING_MAX_CONCURRENCY`

The existing `TRACEMEM_MODEL_MAX_RETRIES` continues to control extraction and
reranking. It no longer limits embedding retries.

## Observability

Retry diagnostics record the attempt number, failure category or upstream status
code, and next delay. They must not include authorization headers, API keys,
request bodies, or response bodies.

## Testing

Automated tests cover:

- Repeated retryable responses followed by success.
- Retry-After handling and capped exponential backoff.
- Immediate failure for non-retryable HTTP errors.
- Connection error recovery.
- A maximum of three simultaneous embedding HTTP attempts.
- Cancellation propagation without further retries.
- Existing dual-header authentication behavior.
