# Embedding Graceful Degradation Design

## Goal

Prevent an upstream embedding rejection or malformed embedding response from
causing Add or Search to return HTTP 502. TraceMem should make a bounded effort
to recover, preserve original memory content when recovery fails, and continue
with lexical retrieval.

The existing persistent retry behavior for temporary failures remains in
place. This design changes only failures that currently exit immediately.

## Scope

The change applies centrally to `OpenAIEmbedder`, so the same behavior covers:

- Episode embeddings during Add.
- Extracted-card embeddings during Add.
- Query embeddings during Search.

The public Add and Search request and response schemas do not change. Database,
validation, authentication, and unexpected local programming errors retain
their existing HTTP behavior.

## Failure Classification

Temporary failures continue retrying indefinitely with the existing capped
exponential backoff:

- HTTP 429, 500, 502, 503, and 504.
- Connection failures, request timeouts, and premature stream termination.

Failures that previously exited immediately receive bounded recovery:

- Other HTTP error responses, including 400, 401, 403, 404, 413, and 422.
- HTTP 200 responses whose embedding body has invalid JSON, missing fields,
  the wrong item count, nonnumeric vectors, empty vectors, or inconsistent
  dimensions.

## Bounded Recovery

For a non-temporary failure, the original batch is attempted at most ten times.
The first request counts as attempt one. Attempts use a fixed 0.5-second delay,
which keeps the additional wait bounded while still satisfying the requested
retry policy.

If a multi-item batch still fails after ten attempts, the embedder makes one
isolation request for each item:

- Items that succeed keep their real embedding.
- Items that receive another non-temporary failure receive an empty,
  zero-dimensional vector.
- Temporary failures during an isolation request retain the existing persistent
  retry behavior.

A single-item batch that still fails after ten attempts receives an empty,
zero-dimensional vector without another isolation request. Returned vectors
remain in input order, and the result count always matches the input count.

This isolates content-specific failures without repeatedly multiplying every
bad request by the batch size. It also preserves valid vectors when one item
causes a larger batch to be rejected.

## Storage and Retrieval Behavior

An empty vector can be stored by the existing schema as a non-null empty BLOB
with `embedding_dimensions=0`. No database migration is required.

Add still commits every original episode, its lexical text, and its FTS entry.
The endpoint returns the normal successful Add response after the commit. A
degraded card is also retained for lexical retrieval and state processing when
the extractor produced a valid card.

Vector retrieval already excludes vectors whose dimensions do not match the
query vector and excludes zero-norm vectors. Empty vectors therefore do not
produce misleading semantic scores.

When a Search query embedding degrades to an empty vector, vector channels
produce no candidates, while episode/card FTS and state channels continue. The
Search endpoint returns its normal response, possibly with fewer or no results,
rather than HTTP 502.

Degraded memories remain retrievable through lexical/BM25 matching. They lose
semantic-vector recall, so queries without useful lexical overlap may not find
them.

## Observability

Embedding diagnostics record enough metadata to distinguish payload rejection,
provider configuration problems, and malformed responses without exposing
evaluation content:

- Upstream HTTP status when available.
- Attempt number and next delay.
- Batch item count.
- Maximum input character count.
- Total input character count.
- Failure category for transport or response-schema errors.
- Whether a batch was split for isolation.
- Count of real and empty vectors returned.

Logs must not include message text, query text, model request or response
bodies, endpoint URLs, authorization headers, API keys, user IDs, or session
IDs.

Suggested stable events are:

- `embedding_retry`: temporary or bounded retry metadata.
- `embedding_batch_isolation`: a ten-attempt batch failure is being split.
- `embedding_degraded`: final real/empty vector counts and safe failure
  metadata.

Add and Search completion logs continue to show HTTP-level success. Their
existing embedding-completed events gain an `empty_vectors` count so the
degradation can be correlated with the owning request without logging content.

## API Error Behavior

Upstream embedding HTTP errors and invalid embedding response bodies no longer
escape as `ModelTransportError` or `ModelResponseError` after bounded recovery.
They return empty vectors and allow normal Add/Search processing.

Unexpected local exceptions are not swallowed. This design does not convert
database corruption, programming errors, invalid public API requests, or
TraceMem authentication failures into successful responses.

## Testing

Automated tests cover:

- A non-temporary HTTP error succeeds on an attempt before the ten-attempt
  limit.
- Ten rejected batch attempts trigger item isolation.
- Isolation preserves good vectors and substitutes empty vectors only for bad
  items while preserving order.
- A rejected single-item batch returns one empty vector after ten attempts.
- Invalid HTTP 200 response bodies use the same bounded recovery path.
- Temporary failures continue retrying persistently and do not consume the
  bounded non-temporary attempt budget.
- Empty episode vectors are committed and remain searchable through FTS.
- An empty Search query vector runs lexical channels and returns HTTP 200.
- Logs include status, attempts, batch size, maximum characters, total
  characters, and empty-vector counts without content, response bodies, URLs,
  or credentials.
- Existing embedding authentication, concurrency, cancellation, Add, Search,
  and observability tests remain green.
