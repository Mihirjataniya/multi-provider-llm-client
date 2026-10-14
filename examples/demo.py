"""
Live tour: real calls to Groq and Gemini.

    uv run examples/demo.py                # keys from ./.env
    LLM_ENV_FILE=path/to/.env uv run examples/demo.py
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from pydantic import BaseModel

from llm_client import AuthError, Client, ProviderError
from llm_client.providers.groq import GroqProvider

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(os.environ.get("LLM_ENV_FILE", ".env"))
logging.basicConfig(level=logging.WARNING, format="  log: %(message)s")

ASK = [{"role": "user", "content": "In one sentence: why do LLM APIs return 429?"}]


def show(r):
    print(f"  provider={r.provider} model={r.model} attempts={r.attempts}")
    print(f"  tokens in={r.input_tokens} out={r.output_tokens} cached={r.cached_input_tokens} "
          f"reasoning={r.reasoning_tokens} finish={r.finish_reason}")
    ttft = f"{r.ttft_ms:.0f}ms" if r.ttft_ms is not None else "n/a"
    cost = f"${r.cost_usd:.6f}" if r.cost_usd is not None else "unpriced"
    print(f"  latency={r.latency_ms:.0f}ms ttft={ttft} cost={cost} request_id={r.request_id}")


class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    po_number: str | None


class Outage(GroqProvider):
    """Groq, but every call fails like a 503. Simulated, to show fallback."""
    async def complete(self, req):
        raise ProviderError("simulated 503", provider=self.name, status_code=503)


async def main():
    client = Client("llm_client.toml")

    for name in client.settings.order:
        print(f"\n1. chat() pinned to {name}")
        r = await client.chat(ASK, provider=name, max_tokens=80, metadata={"demo": "chat"})
        print(f"  → {r.text.strip()}")
        show(r)

    print("\n2. stream() on the primary")
    s = client.stream(ASK, max_tokens=80)
    print("  → ", end="")
    async for text in s:
        print(text, end="", flush=True)
    print()
    show(s.response)

    print("\n3. structured output on each provider")
    doc = "ACME Ltd invoice. Total due: Rs 49,200.00 (INR). Thank you!"
    for name in client.settings.order:
        r = await client.chat([{"role": "user", "content": f"Extract the invoice.\n\n{doc}"}],
                              provider=name, schema=Invoice, temperature=0, max_tokens=200)
        print(f"  {name}: {r.parsed!r}")

    print("\n4. fallback: groq 'down' (simulated 503) → gemini")
    groq_cfg = client.settings.providers["groq"]
    flaky = Client(settings=client.settings, providers={
        "groq": Outage(groq_cfg, "unused"),
        "gemini": client.providers["gemini"],
    })
    r = await flaky.chat(ASK, max_tokens=80)
    show(r)

    print("\n5. bad API key → AuthError, no fallback")
    broken = Client(settings=client.settings, providers={
        "groq": GroqProvider(groq_cfg, "gsk_not_a_real_key"),
        "gemini": client.providers["gemini"],
    })
    try:
        await broken.chat(ASK, max_tokens=20)
    except AuthError as e:
        print(f"  raised {type(e).__name__}: {e}")


asyncio.run(main())
