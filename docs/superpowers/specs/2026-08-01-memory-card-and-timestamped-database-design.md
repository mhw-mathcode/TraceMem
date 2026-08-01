# Memory Card Reliability and Timestamped Database Design

## Goal

Ensure useful memory cards survive minor extractor schema deviations and are
actually persisted and searchable. Isolate each server run by creating one new
timestamped SQLite database at application startup, while sharing that database
across all Add and Search requests handled by the process.

This change improves the reliability of the structured-memory path. It cannot
guarantee a particular evaluation score because extraction quality and the
evaluation corpus remain external variables.

## Current Failure

The extractor accepts only a small fixed set of `kind`, `polarity`, and
`relation_hint` values. The prompt enumerates card kinds but does not fully
describe the allowed polarity and relation values. Validation is performed on
the complete response: one unsupported optional value, such as
`relation_hint="self-reported preference"`, raises `ModelResponseError` and
causes Add to store only the raw episode. Logs then show `cards=0` and
`raw_only=true`, even when the response contains otherwise useful cards.

The downstream path is already present: valid cards are embedded, committed to
`memory_cards` and card FTS/state tables, and searched through card lexical,
card vector, and state channels. The reliability issue is at the extraction
contract boundary rather than in card persistence or retrieval.

## Chosen Approach

Use a constrained prompt together with tolerant, per-card parsing. Prompt-only
validation remains too dependent on model compliance, while an additional LLM
repair request would add latency, cost, and timeout risk. The parser will repair
only safe, non-semantic formatting deviations and will reject malformed cards
independently instead of discarding the whole response.

## Extraction Contract

The extraction prompt will explicitly enumerate the accepted values:

- `kind`: `fact`, `event`, `preference`, `goal`, `plan`, `relationship`, or
  `causal`.
- `polarity`: `positive`, `negative`, or `uncertain`.
- `relation_hint`: `continue`, `update`, `correction`, `conflict`, or `null`.

It will state that descriptive phrases such as `self-reported preference`
belong in `kind` or card text and must not be placed in `relation_hint`.

## Tolerant Card Parsing

Each returned card is parsed independently:

1. Trim whitespace and compare enum values case-insensitively.
2. Preserve the existing `memory` to `fact` kind alias.
3. Treat an unknown `relation_hint` as `null`, because this field is optional
   guidance and the ledger can determine the relation from existing state.
4. Reject only the individual card when a required field is missing, a required
   enum remains unsupported, or source indexes are invalid.
5. Return every valid card even when other cards in the same response are
   rejected.

The parser will not invent semantic content or create a synthetic card when the
model intentionally returns an empty card list. Raw episodes remain the safe
fallback when no valid card is available.

## Card Persistence and Retrieval

No public API shape changes. Valid extracted cards continue through the existing
Add flow:

1. Create the raw episode.
2. Extract and validate cards.
3. Embed each accepted card.
4. Apply ledger state transitions.
5. Commit the episode, cards, FTS rows, vectors, and state in one database
   transaction.

Search continues to fuse episode BM25/vector results with card BM25/vector and
state results. An integration test will prove that a card produced from the
previously failing relation value is stored and can contribute to a Search
result, rather than merely testing the parser in isolation.

## Observability

Extraction logs will include aggregate counts for returned, accepted,
normalized, and rejected cards. No message text, card text, model response body,
user identifier, or credential will be logged.

An unknown optional relation produces a normalization count rather than an Add
degradation. If all cards are invalid, Add still succeeds by storing the raw
episode and records the aggregate rejected count.

## Timestamped Database Lifecycle

Timestamping is enabled by default. At the beginning of the FastAPI lifespan,
the configured database path is resolved once into a new sibling filename. For
example:

```text
data/tracemem.db
  -> data/tracemem-20260801T143052123456Z.db
```

The timestamp is UTC, includes microseconds, and is generated once per process
startup. The resolved path is then used by one `Database` instance for the
entire application lifespan, so Add and Search always see the same memories.
The startup log reports the resolved timestamped path.

The special SQLite path `:memory:` is never rewritten, preserving unit tests and
explicit in-memory use. Existing database files are not renamed, overwritten,
or automatically deleted. Each restart therefore creates a fresh evaluation
database and retains earlier runs for diagnosis.

If the configured filename has an extension, the timestamp is inserted before
it. If it has no extension, the timestamp is appended. A process-id suffix is
used only if the generated timestamped path already exists, preventing an
accidental collision without changing the normal filename format.

## Compatibility and Configuration

The default filesystem behavior changes intentionally from reusing a fixed
database to creating a database per server startup. `TRACEMEM_DATABASE_PATH`
remains the base directory and filename selector. No additional environment
variable is required for the requested default behavior.

SQLite schema initialization still occurs before the application accepts
traffic. A startup failure to create or initialize the resolved file remains a
fatal startup error rather than returning partial service behavior.

## Testing

Automated tests will cover:

- The prompt lists every accepted enum value.
- Case and surrounding whitespace normalization.
- `self-reported preference` in `relation_hint` becomes `null` without losing
  the card.
- One malformed card does not discard valid sibling cards.
- Invalid required enums and source indexes reject only their owning card.
- Aggregate extraction logs contain counts but no user or card content.
- Accepted cards are committed with `raw_only=false`.
- A committed card is available to the existing card retrieval channels.
- A filesystem database path receives one UTC timestamp per application
  startup.
- All requests in one application lifespan share the same resolved database.
- A second startup creates a different database and does not overwrite the
  first.
- `:memory:` remains unchanged.
- Existing Add, Search, retry, authentication, and observability tests remain
  green.
