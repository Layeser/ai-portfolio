"""Thin LLM client. The pipeline depends on the `LLM` protocol only, so tests use a fake."""

from __future__ import annotations

import os
import time
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, user: str, *, model: str | None = None, max_tokens: int = 700) -> str: ...


class GroqLLM:
    """Groq chat completions with back-off on rate limits (free tier: 8k tokens / minute)."""

    def __init__(self, model: str, api_key: str | None = None, max_attempts: int = 6):
        from groq import Groq

        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"], timeout=60, max_retries=0)
        self.model = model
        self.max_attempts = max_attempts

    def complete(self, system: str, user: str, *, model: str | None = None, max_tokens: int = 700) -> str:
        from groq import APIConnectionError, InternalServerError, RateLimitError

        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except (RateLimitError, APIConnectionError, InternalServerError) as exc:
                if attempt == self.max_attempts - 1 or "per day" in str(exc):
                    raise  # daily quota: waiting minutes per call is pointless, fail fast (eval is resumable)
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
                time.sleep(float(retry_after) if retry_after else 5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def classify_injection(self, text: str, model: str) -> float:
        """Llama Prompt Guard 2 returns the probability that `text` is a jailbreak / injection."""
        response = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}])
        return float(response.choices[0].message.content)
