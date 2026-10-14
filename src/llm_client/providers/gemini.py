import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as gt

from ..config import ProviderSettings
from ..errors import ConnectionFailedError, LLMError, RequestTimeoutError
from ..types import Completion, FinishReason, Request
from .base import Provider, parse_retry_after, status_family


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, settings: ProviderSettings, api_key: str):
        self.settings = settings
        # google-genai does not retry unless retry_options is set. We leave
        # it unset on purpose, so every attempt goes through our loop.
        self.client = genai.Client(api_key=api_key)

    def _call_args(self, req: Request) -> dict:
        # Gemini has no "system" role and calls the assistant "model".
        system = "\n\n".join(m["content"] for m in req.messages if m["role"] == "system")
        contents = [
            gt.Content(role="model" if m["role"] == "assistant" else "user",
                       parts=[gt.Part(text=m["content"])])
            for m in req.messages if m["role"] != "system"
        ]
        config = gt.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop_sequences=req.stop,
            http_options=gt.HttpOptions(timeout=int(req.timeout_s * 1000)),
        )
        if self.settings.thinking_level:
            config.thinking_config = gt.ThinkingConfig(thinking_level=self.settings.thinking_level.upper())
        if req.schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = req.schema.model_json_schema()
        return {"model": req.model, "contents": contents, "config": config}

    async def complete(self, req: Request) -> Completion:
        r = await self.client.aio.models.generate_content(**self._call_args(req))
        return _completion(text=r.text or "", resp=r, model=req.model)

    async def stream(self, req: Request):
        stream = await self.client.aio.models.generate_content_stream(**self._call_args(req))
        last = None
        async for chunk in stream:
            last = chunk
            if chunk.text:
                yield chunk.text
        yield _completion(text="", resp=last, model=req.model)

    def map_error(self, exc: Exception) -> LLMError | None:
        if isinstance(exc, genai_errors.APIError):
            headers = exc.response.headers if isinstance(exc.response, httpx.Response) else None
            message = exc.message or str(exc)
            # A 429 that names a per-day quota won't clear in seconds.
            daily = exc.code == 429 and "perday" in str(exc.details).lower().replace(" ", "")
            return status_family(exc.code, message, self.name, parse_retry_after(headers), daily)
        if isinstance(exc, httpx.TimeoutException):
            return RequestTimeoutError(str(exc) or "timed out", provider=self.name)
        if isinstance(exc, httpx.TransportError):
            return ConnectionFailedError(str(exc) or type(exc).__name__, provider=self.name)
        return None


def _completion(text: str, resp, model: str) -> Completion:
    usage = resp.usage_metadata if resp else None
    candidate = resp.candidates[0] if resp and resp.candidates else None
    thoughts = (usage.thoughts_token_count or 0) if usage else 0
    return Completion(
        text=text,
        model=(resp.model_version if resp else None) or model,
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        # Gemini reports thinking separately from the answer, but bills both
        # as output. Add them so output_tokens means the same on every provider.
        output_tokens=((usage.candidates_token_count or 0) + thoughts) if usage else 0,
        cached_input_tokens=(usage.cached_content_token_count or 0) if usage else 0,
        reasoning_tokens=thoughts,
        finish_reason=_finish(candidate.finish_reason if candidate else None),
        request_id=resp.response_id if resp else None,
    )


def _finish(reason) -> FinishReason:
    name = getattr(reason, "name", None) or (str(reason) if reason else "")
    if name == "STOP":
        return "stop"
    if name == "MAX_TOKENS":
        return "length"
    if name in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"):
        return "content_filter"
    if name in ("MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"):
        return "tool_calls"
    return "other"
