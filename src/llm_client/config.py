"""
Settings live in a TOML file, not in code. API keys do not: the file only
names the env var that holds each key, so the file is safe to commit.
"""

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .errors import ConfigError


class Defaults(BaseModel):
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout_s: float = 30.0


class RetrySettings(BaseModel):
    max_attempts: int = Field(3, ge=1)   # per provider, first try included
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    max_retry_after_s: float = 20.0      # a longer Retry-After → fall back instead of waiting


class BreakerSettings(BaseModel):
    failure_threshold: int = 5           # consecutive unhealthy calls before opening
    cooldown_s: float = 30.0


class ProviderSettings(BaseModel):
    api_key_env: str
    model: str
    # Provider-specific knobs. Groq: reasoning_effort. Gemini: thinking_level.
    reasoning_effort: str | None = None
    thinking_level: str | None = None


class Price(BaseModel):
    """US dollars per 1M tokens."""
    input: float
    output: float
    cached_input: float | None = None    # unknown → billed at the full input price


class Settings(BaseModel):
    order: list[str]                     # fallback order, first = primary
    defaults: Defaults = Defaults()
    retry: RetrySettings = RetrySettings()
    circuit_breaker: BreakerSettings = BreakerSettings()
    providers: dict[str, ProviderSettings]
    prices: dict[str, Price] = {}


def load_settings(path: str | Path) -> Settings:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        settings = Settings.model_validate(raw)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except (tomllib.TOMLDecodeError, ValidationError) as e:
        raise ConfigError(f"bad config file {path}: {e}") from None

    missing = [name for name in settings.order if name not in settings.providers]
    if missing:
        raise ConfigError(f"order lists providers with no [providers.*] section: {missing}")
    return settings
