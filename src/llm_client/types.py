"""The shapes that go in and come out."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from pydantic import BaseModel

FinishReason = Literal["stop", "length", "content_filter", "tool_calls", "other"]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class Request:
    """One call, after defaults from the config have been filled in."""
    messages: list[Message]
    model: str
    max_tokens: int
    temperature: float
    timeout_s: float
    top_p: float | None = None
    stop: list[str] | None = None
    schema: type[BaseModel] | None = None


@dataclass
class Completion:
    """What a provider adapter returns. Provider-neutral, no timing or cost yet."""
    text: str
    model: str
    input_tokens: int
    output_tokens: int          # everything billed as output, reasoning included
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: FinishReason = "other"
    request_id: str | None = None


@dataclass
class ChatResponse:
    """What the caller gets back from chat(), and from stream() once it ends."""
    text: str
    provider: str               # who actually answered (may differ after fallback)
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    finish_reason: FinishReason
    request_id: str | None
    latency_ms: float           # wall time of the whole call: retries and waits included
    ttft_ms: float | None       # streaming only; None for chat()
    attempts: int               # total tries across all providers
    cost_usd: float | None      # None when the model has no price in the config
    parsed: BaseModel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
