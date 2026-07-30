# Embedding Dual-Credential Authentication Design

## Goal

Allow an OpenAI-compatible embedding endpoint to require both:

```http
Authorization: Bearer <primary-key>
X-Embedding-Key: <extra-key>
```

The second credential must remain optional so existing providers that only
require bearer authentication continue to work.

## Configuration

Add one optional setting:

```dotenv
TRACEMEM_EMBEDDING_EXTRA_KEY=
```

`TRACEMEM_EMBEDDING_API_KEY` remains the bearer credential. An empty or
whitespace-only extra key means that `X-Embedding-Key` is omitted.

## Components and Data Flow

1. `Settings` reads `TRACEMEM_EMBEDDING_EXTRA_KEY` as
   `embedding_extra_key`.
2. `build_model_bundle()` passes the value only to `OpenAIEmbedder`.
3. `OpenAIEmbedder` always sends the bearer `Authorization` header.
4. `OpenAIEmbedder` adds `X-Embedding-Key` only when the configured value is
   non-empty after trimming.
5. Extractor and reranker clients remain unchanged and never receive the
   extra embedding credential.

The header name is intentionally fixed instead of accepting arbitrary JSON
headers. This keeps configuration simple and prevents accidental credential
forwarding to unrelated model endpoints.

## Compatibility and Error Handling

- Existing configurations without the new variable behave exactly as before.
- `TRACEMEM_EMBEDDING_API_KEY` remains required when embedding mode is
  `openai`.
- `TRACEMEM_EMBEDDING_EXTRA_KEY` is never required by TraceMem. If the
  provider requires it but it is absent or invalid, the provider's HTTP
  authentication error follows the existing `ModelTransportError` path.
- Credentials must not appear in application error messages or logs.

## Documentation

Update `.env.example` and the README model configuration example to show the
optional second credential and explain which HTTP header each variable
produces.

## Tests

Add focused tests using an in-process HTTP transport:

1. An embedding request configured with an extra key sends both
   `Authorization` and `X-Embedding-Key`.
2. An embedding request without an extra key omits `X-Embedding-Key`.
3. A whitespace-only extra key is treated as absent.
4. The response remains parsed into the expected embedding vector.

The tests assert observable HTTP request behavior rather than private
implementation details.
