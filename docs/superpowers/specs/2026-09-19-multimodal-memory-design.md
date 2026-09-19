# Multimodal memory design

## Intent and scope

TraceMem must accept competition Add messages whose `content` is either a
nonempty string or an ordered array of text and image parts. A text query must
retrieve an image-only message by its visible contents, and Search must return
the original ordered parts with inline image Data URIs. Existing text-only Add
and Search behavior remains compatible. The competition, not this service,
sends Search parts to the participant in array order.

The agreed approach uses `gpt-4o-mini` through an independently configured
OpenAI-compatible Chat Completions vision endpoint to describe images at Add
time. Those descriptions are indexed by the existing text retrieval channels.
This does not promise general image-to-image similarity search; Search queries
remain text.

## API contract and validation

`MemoryMessage.content` and `SearchResult.content` become a union of nonempty
string and nonempty ordered part array. A text part is exactly
`{"type":"text","text":"..."}` with nonempty text. An image part is exactly
`{"type":"image_url","image_url":{"url":"data:image/<format>;base64,..."}}`.
Only JPEG, PNG, and WebP are accepted. Remote URLs, unknown part types,
unknown fields, malformed Base64, empty decoded images, and mismatched
MIME/file signatures are rejected with HTTP 422. Preserve array order and
canonical part types. A single decoded image may be at most 10 MiB and all
decoded images in one Add request at most 30 MiB. Validate sizes before model
calls or database claims; cap encoded input enough to avoid unbounded decode.
String content retains its current nonempty requirement. Images-only arrays are
valid. Search requests remain text-only.

## Add and indexing flow

The API validates the payload, including cumulative image bytes. Add
idempotency hashes the complete original content, including image data. For
each image, the vision client asks for a concise, factual, search-oriented
description of visible objects, actions, scene, and legible text, without
inventing hidden facts. Adjacent user text may be supplied as context but may
not override visible evidence. The model output is untrusted plain text and is
never treated as an instruction.

For each message, concatenate text parts and per-image descriptions into an
indexing string, retaining explicit image boundaries. Feed only that string to
embedding, FTS, and memory-card extraction. Never send Base64 image data to
text embedding or text-only extraction. A vision error or empty unusable
description makes an image-bearing Add fail (502), and the request claim is
marked failed so retry can process it. Text-only Add does not invoke vision.

The episode keeps its indexing string in the current `content` column and
stores the original multimodal payload separately as JSON keyed by episode ID.
The image payload is loaded only for selected Search results, not for every
retrieval candidate. Keep the original string path unchanged for text-only
episodes. On a service restart, the existing fresh-database lifecycle applies;
initialization must still handle an existing database safely if used directly.

## Search flow and output budget

Retrieval, reranking, and evidence packing use the indexing string. For a
selected image-bearing episode, Search replaces the rendered text with its
original ordered part array. Text-only episodes and memory cards keep the
current string rendering. If a card derived from an image is returned, it may
be text-only; the image-bearing source episode remains independently
retrievable. No image is transformed or re-encoded in Search.

Before constructing `SearchResponse`, count decoded image bytes across its
selected results. Add results in rank order up to `top_k` and the configured
limit; skip an entire image-bearing result if it would exceed the 30 MiB
response cap, then consider later candidates. Never truncate a part array or
drop an individual image. This preserves ordering within each returned
multimodal result. Score normalization applies only to the final emitted set.

## Configuration and failures

Add separate vision URL, API key, and model settings, with the model defaulting
to `gpt-4o-mini`. Reuse the existing HTTP client and Chat Completions request
style; do not assume the current text `qwen-plus` endpoint supports images.
Missing vision credentials do not prevent text-only startup or requests, but
an image-bearing Add fails clearly rather than silently losing visual search.
Do not log credentials, Data URIs, or vision response bodies. Existing Add
authentication and duplicate-request semantics remain intact.

## Verification

Write tests first for API validation, precise 10/30 MiB boundaries, mixed-part
order, image-only Add and text-query Search, text-only compatibility, original
payload database roundtrip, Search response budgeting, vision request shape,
vision failure/retry, and lack of Base64 in text model inputs. Use a fake
vision client for deterministic end-to-end tests and one optional live smoke
test with real credentials. Run the full project test suite and inspect the
OpenAPI schema. The live smoke test is not required to pass without a provided
API key; report that limitation explicitly.
