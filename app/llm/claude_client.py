"""
LLM client abstraction over the Claude API (Anthropic).

The rest of the app (RAG pipeline, chat service, query handler) only ever
depends on the LLMClient protocol, so the backend can be swapped without
touching any of that code, and tests can inject a fake implementation with no
network access.

Two Anthropic-specific details are handled here rather than leaking outward:

1. The system prompt is a top-level `system` parameter, not a message with
  role="system". Callers pass it via the `system` keyword.
2. `max_tokens` is required on every request. It is a ceiling rather than a
  cost - you are billed for tokens actually generated - so it is set
  generously by default and only trimmed deliberately.
"""

from __future__ import annotations

import logging
from typing import Protocol

import anthropic

logger = logging.getLogger("app.llm.claude_client")

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 16000


class LLMClient(Protocol):
    def chat(self, messages: list[dict], *, system: str | None = None, temperature: float = 0.0, prefill: str | None = None) -> str:
        ...


class ClaudeClient:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    # Send query to Claude API
    def chat(self, messages: list[dict], *, system: str | None = None, temperature: float = 0.0, prefill: str | None = None) -> str:

        if prefill:
            messages = [*messages, {"role": "assistant", "content": prefill}]

        kwargs: dict = {"model": self._model,
                        "max_tokens": self._max_tokens,
                        "temperature": temperature,
                        "messages": messages}
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return (prefill or "") + "".join(block.text for block in response.content if block.type == "text")
