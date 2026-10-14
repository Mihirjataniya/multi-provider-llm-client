"""
Offline tests. Fake providers play back a script of results and errors,
so every retry / fallback rule is checked without a network call.
"""

import asyncio
import json
import logging

import pytest
from pydantic import BaseModel

from llm_client import (AllProvidersFailedError, AuthError, BadRequestError, Client,
                        ConnectionFailedError, ParseError, ProviderError, RateLimitError,
                        Settings)
from llm_client.providers.base import Provider
from llm_client.resilience import CircuitBreaker
from llm_client.types import Completion

MSG = [{"role": "user", "content": "hi"}]


def ok(text="hello", **kw) -> Completion:
    base = dict(text=text, model="m", input_tokens=1000, output_tokens=500,
                finish_reason="stop", request_id="req_1")
    return Completion(**(base | kw))


class Fake(Provider):
    """complete(): pops the next item from `script`; raise it if it's an exception.
    stream(): each script item is a list of text pieces / exceptions."""

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)
        self.calls = 0

    async def complete(self, req):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, req):
        self.calls += 1
        for piece in self.script.pop(0):
            if isinstance(piece, Exception):
                raise piece
            yield piece
        yield ok(text="", output_tokens=3)

    def map_error(self, exc):
        return None


def settings(**retry) -> Settings:
    return Settings.model_validate({
        "order": ["a", "b"],
        "retry": {"max_attempts": 3, "base_delay_s": 0.001, "max_delay_s": 0.002} | retry,
        "circuit_breaker": {"failure_threshold": 2, "cooldown_s": 60},
        "providers": {"a": {"api_key_env": "X", "model": "model-a"},
                      "b": {"api_key_env": "Y", "model": "model-b"}},
        "prices": {"model-a": {"input": 1.0, "output": 2.0, "cached_input": 0.1}},
    })


def client(a, b, **retry) -> Client:
    return Client(settings=settings(**retry), providers={"a": a, "b": b})


def rate_limited(**kw):
    return RateLimitError("slow down", provider="a", status_code=429, **kw)


def server_down(name="a"):
    return ProviderError("overloaded", provider=name, status_code=503)


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff waits instead of sleeping."""
    waited = []

    async def fake_sleep(s):
        waited.append(s)

    monkeypatch.setattr("llm_client.client.asyncio.sleep", fake_sleep)
    return waited


# ---------------------------------------------------------------- chat


async def test_success_fills_every_field():
    c = client(Fake("a", [ok()]), Fake("b", []))
    r = await c.chat(MSG, metadata={"feature": "test"})
    assert (r.text, r.provider, r.attempts) == ("hello", "a", 1)
    assert r.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    assert r.ttft_ms is None                      # chat() can't see the first token
    assert r.metadata == {"feature": "test"}


async def test_cached_tokens_priced_lower():
    c = client(Fake("a", [ok(cached_input_tokens=800)]), Fake("b", []))
    r = await c.chat(MSG)
    assert r.cost_usd == pytest.approx((200 * 1.0 + 800 * 0.1 + 500 * 2.0) / 1e6)


async def test_unknown_price_gives_none_not_zero():
    c = client(Fake("a", [server_down(), server_down(), server_down()]), Fake("b", [ok()]))
    r = await c.chat(MSG)
    assert r.provider == "b" and r.cost_usd is None   # model-b has no price


async def test_429_retried_then_succeeds(sleeps):
    a = Fake("a", [rate_limited(), ok()])
    r = await client(a, Fake("b", [])).chat(MSG)
    assert r.attempts == 2 and a.calls == 2 and len(sleeps) == 1


async def test_retry_after_header_is_obeyed(sleeps):
    a = Fake("a", [rate_limited(retry_after=4.0), ok()])
    await client(a, Fake("b", [])).chat(MSG)
    assert len(sleeps) == 1 and 4.0 <= sleeps[0] <= 4.001   # 4s + base_delay jitter


async def test_long_retry_after_falls_back_instead_of_waiting(sleeps):
    a, b = Fake("a", [rate_limited(retry_after=90.0)]), Fake("b", [ok()])
    r = await client(a, b).chat(MSG)
    assert r.provider == "b" and sleeps == [] and a.calls == 1


async def test_daily_quota_skips_retries(sleeps):
    a, b = Fake("a", [rate_limited(quota_exhausted=True)]), Fake("b", [ok()])
    r = await client(a, b).chat(MSG)
    assert r.provider == "b" and a.calls == 1


async def test_401_crashes_with_no_retry_and_no_fallback():
    a, b = Fake("a", [AuthError("bad key", provider="a", status_code=401)]), Fake("b", [ok()])
    with pytest.raises(AuthError):
        await client(a, b).chat(MSG)
    assert a.calls == 1 and b.calls == 0


async def test_400_crashes_with_no_fallback():
    a, b = Fake("a", [BadRequestError("bad", provider="a", status_code=400)]), Fake("b", [ok()])
    with pytest.raises(BadRequestError):
        await client(a, b).chat(MSG)
    assert b.calls == 0


async def test_fallback_after_retries_exhausted(sleeps):
    a = Fake("a", [server_down(), server_down(), server_down()])
    b = Fake("b", [ok("from b")])
    r = await client(a, b).chat(MSG)
    assert (r.text, r.provider, r.attempts) == ("from b", "b", 4)
    assert a.calls == 3


async def test_all_down_raises_with_every_reason(sleeps):
    a = Fake("a", [server_down()] * 3)
    b = Fake("b", [server_down("b")] * 3)
    with pytest.raises(AllProvidersFailedError) as e:
        await client(a, b).chat(MSG)
    assert set(e.value.errors) == {"a", "b"}


async def test_timeout_is_retryable(sleeps):
    class Slow(Fake):
        async def complete(self, req):
            self.calls += 1
            if self.calls == 1:
                await asyncio.Event().wait()          # hangs forever
            return ok()

    a = Slow("a", [])
    r = await client(a, Fake("b", [])).chat(MSG, timeout_s=0.05)
    assert r.attempts == 2


async def test_unknown_exception_is_not_swallowed():
    a = Fake("a", [KeyError("bug in our code")])
    with pytest.raises(KeyError):
        await client(a, Fake("b", [ok()])).chat(MSG)


async def test_model_override_only_applies_to_first_provider(sleeps):
    seen = []

    class Spy(Fake):
        async def complete(self, req):
            seen.append(req.model)
            return await super().complete(req)

    a = Spy("a", [server_down()] * 3)
    b = Spy("b", [ok()])
    await client(a, b).chat(MSG, model="model-a-big")
    assert seen == ["model-a-big"] * 3 + ["model-b"]


async def test_pinned_provider_never_falls_back(sleeps):
    a, b = Fake("a", [ok()]), Fake("b", [server_down("b")] * 3)
    with pytest.raises(AllProvidersFailedError):
        await client(a, b).chat(MSG, provider="b")
    assert a.calls == 0


# ------------------------------------------------------- structured output


class Person(BaseModel):
    name: str
    age: int


async def test_schema_parsed():
    c = client(Fake("a", [ok('{"name": "Asha", "age": 31}')]), Fake("b", []))
    r = await c.chat(MSG, schema=Person)
    assert r.parsed == Person(name="Asha", age=31)


async def test_truncated_json_says_why():
    c = client(Fake("a", [ok('{"name": "As', finish_reason="length")]), Fake("b", []))
    with pytest.raises(ParseError, match="max_tokens") as e:
        await c.chat(MSG, schema=Person)
    assert e.value.raw_text == '{"name": "As'


# ------------------------------------------------------------------ stream


async def test_stream_yields_text_then_response():
    a = Fake("a", [["Hel", "lo"]])
    s = client(a, Fake("b", [])).stream(MSG)
    pieces = [t async for t in s]
    assert pieces == ["Hel", "lo"]
    assert s.response.text == "Hello"
    assert s.response.ttft_ms is not None and s.response.output_tokens == 3


async def test_stream_retries_before_first_token(sleeps):
    a = Fake("a", [[server_down()], ["ok"]])
    s = client(a, Fake("b", [])).stream(MSG)
    assert [t async for t in s] == ["ok"]
    assert s.response.attempts == 2


async def test_stream_never_retries_after_first_token(sleeps):
    a = Fake("a", [["Hel", ConnectionFailedError("reset", provider="a")], ["unused"]])
    b = Fake("b", [["unused"]])
    s = client(a, b).stream(MSG)
    got = []
    with pytest.raises(ConnectionFailedError):
        async for t in s:
            got.append(t)
    assert got == ["Hel"] and a.calls == 1 and b.calls == 0


async def test_stream_falls_back_before_first_token(sleeps):
    a = Fake("a", [[server_down()]] * 3)
    b = Fake("b", [["from b"]])
    s = client(a, b).stream(MSG)
    assert [t async for t in s] == ["from b"]
    assert s.response.provider == "b"


# --------------------------------------------------------- circuit breaker


async def test_breaker_opens_and_skips_provider(sleeps):
    a = Fake("a", [server_down()] * 6)          # threshold is 2 calls
    b = Fake("b", [ok(), ok(), ok()])
    c = client(a, b)
    await c.chat(MSG)
    await c.chat(MSG)
    assert c.breakers["a"].state == "open"
    calls_before = a.calls
    await c.chat(MSG)
    assert a.calls == calls_before              # a skipped entirely


def test_breaker_half_open_lets_one_call_through():
    now = [0.0]
    br = CircuitBreaker(settings().circuit_breaker, clock=lambda: now[0])
    br.record_failure()
    br.record_failure()
    assert br.state == "open" and not br.allow()
    now[0] = 61
    assert br.state == "half-open"
    assert br.allow() is True
    assert br.allow() is False                  # second caller waits for the trial
    br.record_success()
    assert br.state == "closed"


# ----------------------------------------------------------------- logging


async def test_one_json_log_line_per_call(caplog, sleeps):
    caplog.set_level(logging.INFO, logger="llm_client")
    c = client(Fake("a", [rate_limited(), ok()]), Fake("b", []))
    await c.chat(MSG, metadata={"feature": "summarize"})
    line = json.loads([r for r in caplog.records if r.levelno == logging.INFO][-1].message)
    assert line["attempts"] == 2 and line["metadata"] == {"feature": "summarize"}
    assert line["cost_usd"] > 0
