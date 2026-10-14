# multi-provider-llm-client

![tests](https://github.com/Mihirjataniya/multi-provider-llm-client/actions/workflows/tests.yml/badge.svg)

One async interface over Groq and Gemini. Retries with full-jitter backoff, fallback between providers, a circuit breaker, streaming, structured output, and per-call token and cost logging.

```python
from llm_client import Client

client = Client("llm_client.toml")

r = await client.chat([{"role": "user", "content": "hi"}], metadata={"feature": "greet"})
r.text, r.provider, r.attempts, r.latency_ms, r.cost_usd

s = client.stream([{"role": "user", "content": "hi"}])
async for text in s:
    print(text, end="")
s.response.ttft_ms
```

## Design

**Two calls, two return types.** `chat()` returns a `ChatResponse`. `stream()` yields text, then exposes the same `ChatResponse` on `.response`. No `stream=True` flag whose return type changes.

**Errors come in two families.**

| Family | Errors | What the client does |
|---|---|---|
| Provider unhealthy (`RetryableError`) | `RateLimitError` 429, `ProviderError` 5xx, `RequestTimeoutError`, `ConnectionFailedError` | retry with backoff, then fall back to the next provider |
| Our fault | `AuthError` 401/403, `BadRequestError` 400/404/422, `ParseError` | raise immediately. No retry, no fallback |

Fallback is for a broken provider. A bad key or a bad request would fail on the next provider too, and falling back would hide the bug.

**Retries.**
- Full-jitter exponential backoff.
- A `Retry-After` header is obeyed, unless it's longer than `max_retry_after_s`, in which case the client falls back instead.
- A 429 that names a daily quota skips straight to fallback.
- Both SDKs' own retries are switched off, so every attempt is counted and logged.

**Streaming** retries only before the first token. Once text has reached the caller, a failure is raised rather than retried, because a retry would repeat text the caller already has.

**Circuit breaker** per provider. After `failure_threshold` consecutive failed calls, the provider is skipped for `cooldown_s`. Then one trial call is let through (half-open).

**Settings** are in `llm_client.toml`: fallback order, defaults, retry, breaker, models, prices. Keys are not. The file only names the env var for each key.

**Every call logs one JSON line** to the `llm_client` logger: provider, model, attempts, latency, TTFT, input/output/cached/reasoning tokens, cost, finish reason, request ID, and your `metadata`. Retries show up in `attempts` instead of hiding inside a latency number.

## Token accounting

`output_tokens` means the same thing on every provider: everything billed as output, reasoning included. Groq already counts reasoning inside `completion_tokens`. Gemini reports `thoughts_token_count` separately, so the client adds it.

`cost_usd` prices uncached input, cached input and output separately. It's `None`, not `0`, when the model has no price in the config.

## Layout

```
src/llm_client/
├── client.py         Client.chat() / Client.stream(): retry loop, fallback, cost, logging
├── resilience.py     full-jitter backoff, Retry-After handling, circuit breaker
├── errors.py         the two error families
├── types.py          Message → Request → Completion → ChatResponse
├── config.py         TOML settings, validated with Pydantic
├── schema.py         Pydantic model → strict JSON schema
└── providers/
    ├── base.py       adapter contract + shared HTTP-status mapping
    ├── groq.py       Groq SDK adapter
    └── gemini.py     google-genai SDK adapter
```

Adding a provider means writing one adapter. It translates the request, the response and the errors. Retries, fallback, timing and cost are already handled in the client.

## Known limits

- The stream timeout applies **per chunk**, so it catches a stream that goes silent. A stream that keeps trickling slowly has no total cap.
- The circuit breaker state is in memory, per process. Several workers don't share it.
- Text messages only. Tool calling and images aren't supported yet.

## Run

```bash
uv sync
uv run pytest                    # 48 offline tests, no API keys needed
cp .env.example .env             # add GROQ_API_KEY, GEMINI_API_KEY
uv run examples/demo.py          # live calls
```

Prices in `llm_client.toml` were checked on 2026-10-12. `qwen/qwen3.8-27b` is a Groq preview model. Check the pricing pages before relying on the numbers.
