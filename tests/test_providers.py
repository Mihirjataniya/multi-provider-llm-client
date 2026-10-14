"""
Adapter tests: real SDK exception and response objects, no network.
Checks the translation layer: SDK error → our error, SDK usage → our tokens.
"""

from types import SimpleNamespace as NS

import groq
import httpx
import pytest
from google.genai import errors as genai_errors

from llm_client import (AuthError, BadRequestError, ConnectionFailedError, ProviderError,
                        RateLimitError, RequestTimeoutError)
from llm_client.config import ProviderSettings
from llm_client.providers.base import parse_retry_after, status_family
from llm_client.providers.gemini import GeminiProvider
from llm_client.providers.gemini import _completion as gemini_completion
from llm_client.providers.groq import GroqProvider
from llm_client.providers.groq import _completion as groq_completion
from llm_client.types import Request

REQ = httpx.Request("POST", "https://example.test")


def groq_status(cls, code, message="err", headers=None):
    return cls(message, response=httpx.Response(code, headers=headers or {}, request=REQ), body=None)


@pytest.fixture
def groq_p():
    return GroqProvider(ProviderSettings(api_key_env="X", model="m"), "test-key")


@pytest.fixture
def gemini_p():
    return GeminiProvider(ProviderSettings(api_key_env="Y", model="m"), "test-key")


# ------------------------------------------------------------- shared rules


@pytest.mark.parametrize("code,cls", [
    (429, RateLimitError), (401, AuthError), (403, AuthError), (408, RequestTimeoutError),
    (400, BadRequestError), (404, BadRequestError), (422, BadRequestError),
    (500, ProviderError), (503, ProviderError),
])
def test_status_family(code, cls):
    assert type(status_family(code, "x", "p", None)) is cls


def test_retry_after_parsing():
    assert parse_retry_after({"retry-after": "3"}) == 3.0
    assert parse_retry_after({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}) is None
    assert parse_retry_after({}) is None
    assert parse_retry_after(None) is None


# --------------------------------------------------------------------- groq


def test_groq_429_keeps_retry_after(groq_p):
    err = groq_p.map_error(groq_status(groq.RateLimitError, 429, headers={"retry-after": "7"}))
    assert isinstance(err, RateLimitError) and err.retry_after == 7.0 and not err.quota_exhausted


def test_groq_daily_quota_detected(groq_p):
    exc = groq_status(groq.RateLimitError, 429, "Rate limit reached on tokens per day (TPD)")
    assert groq_p.map_error(exc).quota_exhausted


def test_groq_timeout_before_connection(groq_p):
    # APITimeoutError subclasses APIConnectionError; the order of checks matters.
    assert isinstance(groq_p.map_error(groq.APITimeoutError(request=REQ)), RequestTimeoutError)
    assert isinstance(groq_p.map_error(groq.APIConnectionError(request=REQ)), ConnectionFailedError)


def test_groq_auth_and_5xx(groq_p):
    assert isinstance(groq_p.map_error(groq_status(groq.AuthenticationError, 401)), AuthError)
    assert isinstance(groq_p.map_error(groq_status(groq.InternalServerError, 502)), ProviderError)


def test_groq_unknown_exception_passes_through(groq_p):
    assert groq_p.map_error(ValueError("bug")) is None


def test_groq_sdk_retries_disabled(groq_p):
    assert groq_p.client.max_retries == 0


def test_groq_usage_normalised():
    usage = NS(prompt_tokens=100, completion_tokens=40,
               prompt_tokens_details=NS(cached_tokens=60),
               completion_tokens_details=NS(reasoning_tokens=15))
    c = groq_completion("hi", "m", usage, "length", "req_1")
    assert (c.input_tokens, c.output_tokens, c.cached_input_tokens, c.reasoning_tokens) == (100, 40, 60, 15)
    assert c.finish_reason == "length"


def test_groq_unknown_finish_reason_is_other():
    assert groq_completion("", "m", None, "function_call", None).finish_reason == "other"


# ------------------------------------------------------------------- gemini


def gemini_error(code, details=None, headers=None):
    body = {"error": {"code": code, "message": "err", "status": "X", "details": details or []}}
    return genai_errors.APIError(code, body, httpx.Response(code, headers=headers or {}, request=REQ))


def test_gemini_429_and_retry_after(gemini_p):
    err = gemini_p.map_error(gemini_error(429, headers={"retry-after": "5"}))
    assert isinstance(err, RateLimitError) and err.retry_after == 5.0 and not err.quota_exhausted


def test_gemini_daily_quota_detected(gemini_p):
    details = [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]
    assert gemini_p.map_error(gemini_error(429, details)).quota_exhausted


def test_gemini_status_codes(gemini_p):
    assert isinstance(gemini_p.map_error(gemini_error(400)), BadRequestError)
    assert isinstance(gemini_p.map_error(gemini_error(403)), AuthError)
    assert isinstance(gemini_p.map_error(gemini_error(503)), ProviderError)


def test_gemini_transport_errors(gemini_p):
    assert isinstance(gemini_p.map_error(httpx.ReadTimeout("slow", request=REQ)), RequestTimeoutError)
    assert isinstance(gemini_p.map_error(httpx.ConnectError("refused", request=REQ)), ConnectionFailedError)
    assert gemini_p.map_error(KeyError("bug")) is None


def test_gemini_messages_translated(gemini_p):
    req = Request(messages=[{"role": "system", "content": "be brief"},
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"}],
                  model="m", max_tokens=10, temperature=0.0, timeout_s=5)
    args = gemini_p._call_args(req)
    assert args["config"].system_instruction == "be brief"
    assert [c.role for c in args["contents"]] == ["user", "model"]   # no system role in contents
    assert args["config"].http_options.timeout == 5000                # milliseconds


def test_gemini_thinking_tokens_billed_as_output():
    resp = NS(usage_metadata=NS(prompt_token_count=100, candidates_token_count=30,
                                thoughts_token_count=20, cached_content_token_count=None),
              candidates=[NS(finish_reason=NS(name="MAX_TOKENS"))],
              model_version="m-001", response_id="r1")
    c = gemini_completion("hi", resp, "m")
    assert (c.output_tokens, c.reasoning_tokens, c.cached_input_tokens) == (50, 20, 0)
    assert c.finish_reason == "length" and c.model == "m-001"
