import groq

from ..config import ProviderSettings
from ..errors import ConnectionFailedError, LLMError, RequestTimeoutError
from ..schema import strict_json_schema
from ..types import Completion, FinishReason, Request
from .base import Provider, parse_retry_after, status_family


class GroqProvider(Provider):
    name = "groq"

    def __init__(self, settings: ProviderSettings, api_key: str):
        self.settings = settings
        # max_retries=0: the SDK would otherwise retry 2x silently, nested
        # inside our retries, and those attempts would never reach our logs.
        self.client = groq.AsyncGroq(api_key=api_key, max_retries=0)

    def _params(self, req: Request) -> dict:
        params = {
            "model": req.model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "timeout": req.timeout_s,
        }
        if req.top_p is not None:
            params["top_p"] = req.top_p
        if req.stop:
            params["stop"] = req.stop
        if self.settings.reasoning_effort:
            params["reasoning_effort"] = self.settings.reasoning_effort
        if req.schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": req.schema.__name__,
                    "schema": strict_json_schema(req.schema),
                    "strict": True,
                },
            }
        return params

    async def complete(self, req: Request) -> Completion:
        r = await self.client.chat.completions.create(**self._params(req))
        choice = r.choices[0]
        return _completion(
            text=choice.message.content or "",
            model=r.model,
            usage=r.usage,
            finish=choice.finish_reason,
            request_id=r.x_groq.id if r.x_groq else r.id,
        )

    async def stream(self, req: Request):
        stream = await self.client.chat.completions.create(**self._params(req), stream=True)
        finish = None
        usage = None
        model = req.model
        request_id = None
        async for chunk in stream:
            model = chunk.model or model
            if chunk.x_groq:
                request_id = chunk.x_groq.id or request_id
                usage = chunk.x_groq.usage or usage      # final chunk only
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.content:
                    yield choice.delta.content
                finish = choice.finish_reason or finish
        yield _completion(text="", model=model, usage=usage, finish=finish, request_id=request_id)

    def map_error(self, exc: Exception) -> LLMError | None:
        # Order matters: APITimeoutError is a subclass of APIConnectionError.
        if isinstance(exc, groq.APITimeoutError):
            return RequestTimeoutError(str(exc), provider=self.name)
        if isinstance(exc, groq.APIConnectionError):
            return ConnectionFailedError(str(exc), provider=self.name)
        if isinstance(exc, groq.APIStatusError):
            retry_after = parse_retry_after(exc.response.headers)
            return status_family(exc.status_code, exc.message, self.name, retry_after,
                                 quota_exhausted=_is_daily_quota(exc))
        return None


def _is_daily_quota(exc: "groq.APIStatusError") -> bool:
    # Groq's 429 body names the limit that was hit, e.g. "tokens per day (TPD)".
    text = str(exc).lower()
    return exc.status_code == 429 and ("per day" in text or "(rpd)" in text or "(tpd)" in text)


def _completion(text, model, usage, finish, request_id) -> Completion:
    cached = reasoning = 0
    if usage and usage.prompt_tokens_details:
        cached = usage.prompt_tokens_details.cached_tokens or 0
    if usage and usage.completion_tokens_details:
        reasoning = usage.completion_tokens_details.reasoning_tokens or 0
    return Completion(
        text=text,
        model=model,
        input_tokens=usage.prompt_tokens if usage else 0,
        # OpenAI-style APIs already count reasoning inside completion_tokens.
        output_tokens=usage.completion_tokens if usage else 0,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
        finish_reason=_finish(finish),
        request_id=request_id,
    )


def _finish(reason: str | None) -> FinishReason:
    if reason in ("stop", "length", "content_filter", "tool_calls"):
        return reason
    return "other"
