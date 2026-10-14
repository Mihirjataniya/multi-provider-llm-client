"""
The one class callers use.

    client = Client("llm_client.toml")
    r = await client.chat([{"role": "user", "content": "hi"}])

    s = client.stream([{"role": "user", "content": "hi"}])
    async for text in s:
        print(text, end="")
    s.response.cost_usd
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from .config import Settings, load_settings
from .errors import (AllProvidersFailedError, ConfigError, LLMError, ParseError,
                     ProviderError, RequestTimeoutError, RetryableError)
from .providers.base import Provider
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider
from .resilience import CircuitBreaker, next_delay
from .types import ChatResponse, Completion, Message, Request

log = logging.getLogger("llm_client")

PROVIDER_CLASSES: dict[str, type[Provider]] = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


class Client:
    def __init__(self, config: str | Path = "llm_client.toml", *,
                 settings: Settings | None = None,
                 providers: dict[str, Provider] | None = None):
        self.settings = settings or load_settings(config)
        # `providers` lets tests swap in fakes. Normally built from the config.
        self.providers = providers or self._build_providers()
        self.breakers = {name: CircuitBreaker(self.settings.circuit_breaker)
                         for name in self.settings.order}

    def _build_providers(self) -> dict[str, Provider]:
        built = {}
        for name in self.settings.order:
            cfg = self.settings.providers[name]
            if name not in PROVIDER_CLASSES:
                raise ConfigError(f"unknown provider '{name}'. Known: {list(PROVIDER_CLASSES)}")
            key = os.environ.get(cfg.api_key_env)
            if not key:
                # Fail at startup, not on the first request in production.
                raise ConfigError(f"env var {cfg.api_key_env} is not set (needed by '{name}')")
            built[name] = PROVIDER_CLASSES[name](cfg, key)
        return built

    # ------------------------------------------------------------------ chat

    async def chat(self, messages: list[Message], *,
                   model: str | None = None,
                   provider: str | None = None,
                   max_tokens: int | None = None,
                   temperature: float | None = None,
                   top_p: float | None = None,
                   stop: list[str] | None = None,
                   schema: type[BaseModel] | None = None,
                   timeout_s: float | None = None,
                   metadata: dict[str, Any] | None = None) -> ChatResponse:
        start = time.perf_counter()
        attempts = 0
        errors: dict[str, LLMError] = {}
        overrides = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                         top_p=top_p, stop=stop, schema=schema, timeout_s=timeout_s)

        for i, name in enumerate(self._chain(provider)):
            breaker = self.breakers[name]
            if not breaker.allow():
                errors[name] = ProviderError("circuit open, skipped", provider=name)
                continue
            p = self.providers[name]
            req = self._request(name, messages, primary=(i == 0), **overrides)

            for attempt in range(self.settings.retry.max_attempts):
                attempts += 1
                try:
                    async with asyncio.timeout(req.timeout_s):
                        comp = await p.complete(req)
                except Exception as exc:
                    err = self._map(p, exc)
                    if not isinstance(err, RetryableError):
                        breaker.trial_in_flight = False
                        self._log_failure(name, req, attempts, start, err, metadata)
                        raise err from exc               # our bug: crash, no fallback
                    errors[name] = err
                    delay = next_delay(err, attempt, self.settings.retry)
                    if delay is None:
                        break                            # give up on this provider
                    log.warning("retry %s attempt=%d in %.2fs: %s", name, attempt + 1, delay, err)
                    await asyncio.sleep(delay)
                    continue

                breaker.record_success()
                resp = self._response(name, req, comp, start, attempts, None, metadata)
                if schema is not None:
                    resp.parsed = self._parse(schema, resp)
                self._log_success(resp)
                return resp

            breaker.record_failure()
            log.warning("falling back from %s after %d attempt(s): %s", name, attempt + 1, errors[name])

        err = AllProvidersFailedError(errors)
        self._log_failure(None, None, attempts, start, err, metadata)
        raise err

    # ---------------------------------------------------------------- stream

    def stream(self, messages: list[Message], **kwargs) -> "Stream":
        """Same arguments as chat(). Iterate it for text; read .response after."""
        return Stream(self._stream(messages, **kwargs))

    async def _stream(self, messages, *, model=None, provider=None, max_tokens=None,
                      temperature=None, top_p=None, stop=None, schema=None,
                      timeout_s=None, metadata=None):
        start = time.perf_counter()
        attempts = 0
        errors: dict[str, LLMError] = {}
        overrides = dict(model=model, max_tokens=max_tokens, temperature=temperature,
                         top_p=top_p, stop=stop, schema=schema, timeout_s=timeout_s)

        for i, name in enumerate(self._chain(provider)):
            breaker = self.breakers[name]
            if not breaker.allow():
                errors[name] = ProviderError("circuit open, skipped", provider=name)
                continue
            p = self.providers[name]
            req = self._request(name, messages, primary=(i == 0), **overrides)

            for attempt in range(self.settings.retry.max_attempts):
                attempts += 1
                first_token_at = None
                parts: list[str] = []
                final: Completion | None = None
                chunks = aiter(p.stream(req))
                try:
                    while True:
                        try:
                            # Timeout per chunk: catches a stream that goes silent.
                            async with asyncio.timeout(req.timeout_s):
                                item = await anext(chunks)
                        except StopAsyncIteration:
                            break
                        if isinstance(item, Completion):
                            final = item
                            continue
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        parts.append(item)
                        yield item
                except Exception as exc:
                    err = self._map(p, exc)
                    if first_token_at is not None or not isinstance(err, RetryableError):
                        # Tokens already reached the caller. A retry would
                        # repeat them, so the stream fails here, no fallback.
                        if isinstance(err, RetryableError):
                            breaker.record_failure()
                        else:
                            breaker.trial_in_flight = False
                        self._log_failure(name, req, attempts, start, err, metadata)
                        raise err from exc
                    errors[name] = err
                    delay = next_delay(err, attempt, self.settings.retry)
                    if delay is None:
                        break
                    log.warning("retry %s stream attempt=%d in %.2fs: %s", name, attempt + 1, delay, err)
                    await asyncio.sleep(delay)
                    continue
                finally:
                    await chunks.aclose()

                breaker.record_success()
                final = final or Completion(text="", model=req.model, input_tokens=0, output_tokens=0)
                final.text = "".join(parts)
                ttft = (first_token_at - start) * 1000 if first_token_at else None
                resp = self._response(name, req, final, start, attempts, ttft, metadata)
                if schema is not None:
                    resp.parsed = self._parse(schema, resp)
                self._log_success(resp)
                yield resp
                return

            breaker.record_failure()
            log.warning("falling back from %s after %d attempt(s): %s", name, attempt + 1, errors[name])

        err = AllProvidersFailedError(errors)
        self._log_failure(None, None, attempts, start, err, metadata)
        raise err

    # --------------------------------------------------------------- helpers

    def _chain(self, provider: str | None) -> list[str]:
        if provider is None:
            return self.settings.order
        if provider not in self.providers:
            raise ConfigError(f"unknown provider '{provider}'")
        return [provider]                     # pinned: no fallback

    def _request(self, name, messages, *, primary, model, max_tokens, temperature,
                 top_p, stop, schema, timeout_s) -> Request:
        d = self.settings.defaults
        # A model override names one provider's model. It applies to the first
        # provider tried; fallback providers use their own configured model.
        chosen = model if (model and primary) else self.settings.providers[name].model
        return Request(
            messages=messages,
            model=chosen,
            max_tokens=max_tokens if max_tokens is not None else d.max_tokens,
            temperature=temperature if temperature is not None else d.temperature,
            timeout_s=timeout_s if timeout_s is not None else d.timeout_s,
            top_p=top_p,
            stop=stop,
            schema=schema,
        )

    @staticmethod
    def _map(p: Provider, exc: Exception) -> LLMError:
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, TimeoutError):     # from our asyncio.timeout
            return RequestTimeoutError("no response within timeout", provider=p.name)
        mapped = p.map_error(exc)
        if mapped is None:
            raise exc                         # not a provider error: a real bug, surface it
        return mapped

    def _response(self, name, req, comp: Completion, start, attempts, ttft_ms, metadata) -> ChatResponse:
        return ChatResponse(
            text=comp.text,
            provider=name,
            model=comp.model,
            input_tokens=comp.input_tokens,
            output_tokens=comp.output_tokens,
            cached_input_tokens=comp.cached_input_tokens,
            reasoning_tokens=comp.reasoning_tokens,
            finish_reason=comp.finish_reason,
            request_id=comp.request_id,
            latency_ms=(time.perf_counter() - start) * 1000,
            ttft_ms=ttft_ms,
            attempts=attempts,
            # Priced by the model we asked for. The served name can carry a
            # version suffix that isn't in the price table.
            cost_usd=self._cost(req.model, comp),
            metadata=metadata or {},
        )

    def _cost(self, model: str, comp: Completion) -> float | None:
        price = self.settings.prices.get(model)
        if price is None:
            return None
        cached_price = price.cached_input if price.cached_input is not None else price.input
        uncached = comp.input_tokens - comp.cached_input_tokens
        return (uncached * price.input
                + comp.cached_input_tokens * cached_price
                + comp.output_tokens * price.output) / 1_000_000

    @staticmethod
    def _parse(schema: type[BaseModel], resp: ChatResponse) -> BaseModel:
        try:
            return schema.model_validate_json(resp.text)
        except ValidationError as e:
            hint = " (output was cut off: finish_reason=length, raise max_tokens)" \
                if resp.finish_reason == "length" else ""
            raise ParseError(f"response failed {schema.__name__}{hint}: {e.error_count()} error(s)",
                             raw_text=resp.text, provider=resp.provider) from e

    # --------------------------------------------------------------- logging
    # One JSON line per call. Retries live in `attempts`, so they can't hide
    # inside a latency number.

    @staticmethod
    def _log_success(r: ChatResponse) -> None:
        log.info(json.dumps({
            "event": "llm_call", "ok": True, "provider": r.provider, "model": r.model,
            "attempts": r.attempts, "latency_ms": round(r.latency_ms, 1),
            "ttft_ms": round(r.ttft_ms, 1) if r.ttft_ms is not None else None,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "cached_input_tokens": r.cached_input_tokens, "reasoning_tokens": r.reasoning_tokens,
            "cost_usd": r.cost_usd, "finish_reason": r.finish_reason,
            "request_id": r.request_id, "metadata": r.metadata,
        }, default=str))

    @staticmethod
    def _log_failure(name, req, attempts, start, err: LLMError, metadata) -> None:
        log.error(json.dumps({
            "event": "llm_call", "ok": False, "provider": name,
            "model": req.model if req else None, "attempts": attempts,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "error": type(err).__name__, "message": str(err), "metadata": metadata or {},
        }, default=str))


class Stream:
    """Async-iterate for text deltas. After the loop ends, .response holds the ChatResponse."""

    def __init__(self, gen: AsyncIterator):
        self._gen = gen
        self.response: ChatResponse | None = None

    async def __aiter__(self):
        async for item in self._gen:
            if isinstance(item, ChatResponse):
                self.response = item
            else:
                yield item
