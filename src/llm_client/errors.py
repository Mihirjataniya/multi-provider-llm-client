"""
Every failure the client can raise.

Two families under one base:

    LLMError
    ├── RetryableError          provider is unhealthy → retry, then fall back
    │   ├── RateLimitError      429
    │   ├── RequestTimeoutError no answer in time
    │   ├── ConnectionFailedError  DNS / TCP / TLS / reset
    │   └── ProviderError       5xx
    ├── AuthError               401 / 403    → crash, fix your key
    ├── BadRequestError         400/404/422  → crash, fix your request
    ├── ParseError              output failed the schema
    ├── ConfigError             bad config file or missing env var
    └── AllProvidersFailedError every provider tried, all unhealthy

The rule: only RetryableError is retried or falls back. Everything else is
our bug, and fallback would hide it.
"""


class LLMError(Exception):
    def __init__(self, message: str, *, provider: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        where = f"[{self.provider}] " if self.provider else ""
        code = f" (HTTP {self.status_code})" if self.status_code else ""
        return f"{where}{super().__str__()}{code}"


# --- provider unhealthy: retry, then fall back -----------------------------

class RetryableError(LLMError):
    pass


class RateLimitError(RetryableError):
    def __init__(self, message: str, *, retry_after: float | None = None,
                 quota_exhausted: bool = False, **kw):
        super().__init__(message, **kw)
        # Seconds the provider asked us to wait (Retry-After header), if it said.
        self.retry_after = retry_after
        # A daily quota is gone. Waiting seconds won't help, so skip straight
        # to the next provider instead of burning retries.
        self.quota_exhausted = quota_exhausted


class RequestTimeoutError(RetryableError):
    pass


class ConnectionFailedError(RetryableError):
    pass


class ProviderError(RetryableError):
    pass


# --- our fault: never retried, never falls back ----------------------------

class AuthError(LLMError):
    pass


class BadRequestError(LLMError):
    pass


class ParseError(LLMError):
    def __init__(self, message: str, *, raw_text: str, **kw):
        super().__init__(message, **kw)
        self.raw_text = raw_text


class ConfigError(LLMError):
    pass


class AllProvidersFailedError(LLMError):
    def __init__(self, errors: dict[str, LLMError]):
        self.errors = errors
        detail = "; ".join(f"{name}: {err}" for name, err in errors.items())
        super().__init__(f"all providers failed — {detail}")
