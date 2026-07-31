# Internal Observability Design

## Goal

Add safe, grep-friendly internal logs that show where Add and Search spend time
and where failures occur, while keeping evaluation content and credentials out
of the log stream.

## Output and Configuration

TraceMem writes application logs to standard error. Uvicorn, Docker, systemd,
or the current `nohup` command remains responsible for collecting standard
error. The application does not open or rotate a log file, avoiding new file
permission and lifecycle concerns.

`TRACEMEM_LOG_LEVEL` controls TraceMem application logging and defaults to
`INFO`. Accepted values are `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.
Startup configures the `tracemem` logger exactly once with a concise formatter:

```text
2026-07-31T12:34:56+0000 INFO tracemem.add_service event=add_started request_id=req-1 messages=20
```

Third-party logging remains under Uvicorn's configuration.

## Event Format

Messages use whitespace-separated `key=value` fields with a stable `event`
name. String field values are encoded so whitespace, newlines, and control
characters cannot create ambiguous or forged log lines. Durations are integer
milliseconds measured with a monotonic clock.

The implementation provides a small observability module responsible for:

- Idempotent TraceMem logger configuration.
- Safe scalar field rendering.
- Structured event emission.
- Millisecond duration calculation.

Call sites choose explicit fields; the helper never accepts request objects,
model payloads, headers, or arbitrary mappings.

## Privacy Boundary

Allowed fields include:

- Add `request_id`.
- Generated Search `search_id`.
- Counts, booleans, mode names, status codes, error class names, stage names,
  retry attempts, delays, database path and database byte size.

The following values must never be logged:

- Message content, query text, options text, or extracted card text.
- User IDs or session IDs.
- API keys, authorization headers, or other credentials.
- Model request or response bodies.
- Full exception messages from remote clients when they may contain URLs or
  payload details.

Known safe local database exceptions may be represented by their class name;
the traceback remains available for unexpected internal failures.

## Startup Events

The FastAPI lifespan emits:

- `startup_started`: profile and enabled model modes.
- `database_initialized`: resolved database path, byte size after schema
  initialization, and initialization duration.
- `startup_completed`: total startup duration.
- `shutdown_completed`: shutdown duration.

Startup failures emit `startup_failed` with a stage and exception class before
being re-raised so Uvicorn still refuses to start.

## Add Events

Every Add operation uses the supplied `request_id` as its correlation value.
It emits:

- `add_started`: message count.
- `add_claimed`: claim status, wait attempts, and claim duration.
- `add_episode_embedding_completed`: vector count and duration.
- `add_extraction_completed`: extracted card count and duration.
- `add_card_embedding_completed`: vector count and duration when cards exist.
- `add_degraded`: stage and exception class when extraction/card generation
  safely falls back to raw episode storage.
- `add_commit_completed`: episode count, card count, `raw_only`, and duration.
- `add_completed`: total duration.
- `add_failed`: stage, exception class, and total duration before the exception
  is propagated or converted to the existing API error.

Idempotent completed requests still emit `add_completed` with
`replayed=true`.

## Search Events

Each Search operation generates a random 12-character hexadecimal `search_id`.
It emits:

- `search_started`: requested `top_k` and option count.
- `search_query_embedding_completed`: vector count and duration.
- `search_channels_completed`: per-channel candidate counts and retrieval
  duration.
- `search_retrieved`: fused candidate count and duration.
- `search_reranked`: reranked candidate count and duration.
- `search_degraded`: stage and exception class when reranking safely falls back
  to original scores.
- `search_completed`: final result count and total duration.
- `search_failed`: stage, exception class, and total duration before propagation.

`HybridRetriever.retrieve` accepts the internal `search_id` as an optional
keyword-only parameter so its embedding and channel logs correlate with the
owning Search request. The public HTTP contract does not change.

## Embedding Retry Events

The existing retry warnings are converted to the shared event formatter:

- `embedding_retry`: attempt, failure type or upstream status, and delay.

They continue to omit URLs, headers, bodies, model inputs, and credentials.

## Error Handling

Logging must never change the existing API result or fallback behavior. Log
formatting failures are prevented by accepting only explicitly supported scalar
field types. Application exceptions are logged at the closest useful stage and
then follow the existing control flow:

- Episode embedding failure marks Add failed and returns the existing 502.
- Extraction and reranking model failures retain their existing degradation.
- Database and unexpected internal failures are logged and re-raised.

## Testing

Automated tests cover:

- Logger configuration is idempotent and respects the configured level.
- Structured values cannot inject new log lines.
- Startup records the resolved database path and initialized nonzero size.
- Successful Add logs stage counts and timings.
- Add degradation identifies its stage without logging message content.
- Successful Search correlates retrieval stages with one `search_id`.
- Search degradation identifies reranking without logging query/options text.
- Embedding retries retain their status/error, attempt, and delay fields.
- API keys, authorization values, message content, query text, options, user ID,
  session ID, and model response data are absent from captured logs.
- Existing API behavior and model authentication tests remain green.

