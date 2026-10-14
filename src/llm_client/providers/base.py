"""
The contract every provider adapter follows.

An adapter does three things and nothing else:
  1. translate our Request into the provider's SDK call
  2. translate the SDK's answer into our Completion
  3. translate the SDK's exceptions into our LLMError family

Retries, fallback, timing, cost and logging live in the client, once,
so every provider gets the same behaviour.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..errors import (AuthError, BadRequestError, LLMError, ProviderError,
                      RateLimitError, RequestTimeoutError)
from ..types import Completion, Request


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, req: Request) -> Completion: ...

    @abstractmethod
    def stream(self, req: Request) -> AsyncIterator[str | Completion]:
        """Yield text deltas as they arrive, then one final Completion
        carrying usage and finish_reason (its text may be empty)."""

    @abstractmethod
    def map_error(self, exc: Exception) -> LLMError | None:
        """Turn an SDK exception into ours. None means 'not an SDK error,
        let it propagate as-is' (a real bug in our code)."""


def status_family(code: int, message: str, provider: str, retry_after: float | None,
                  quota_exhausted: bool = False) -> LLMError:
    """HTTP status → our error class. Shared by every adapter."""
    kw = {"provider": provider, "status_code": code}
    if code == 429:
        return RateLimitError(message, retry_after=retry_after, quota_exhausted=quota_exhausted, **kw)
    if code in (401, 403):
        return AuthError(message, **kw)
    if code == 408:
        return RequestTimeoutError(message, **kw)
    if code >= 500:
        return ProviderError(message, **kw)
    return BadRequestError(message, **kw)


def parse_retry_after(headers) -> float | None:
    """Retry-After in seconds. Ignores the rare HTTP-date form."""
    if headers is None:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
