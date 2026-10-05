"""
Shared helper for guardrails that ask an LLM to return a structured verdict (such as trajectory.py, hallucination.py, policy.py).

JSON-parsing and validation-failure handling lives in one place here.
Each guardrail's own check() still decides what "failed to verify" means for it.
This project's chosen posture is fail-closed everywhere a judge is used.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.claude_client import LLMClient

logger = logging.getLogger("app.guardrails.judge_utils")

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class JudgeVerificationError(Exception):
    """Raised when a judge call fails outright or its response can't be parsed into the expected verdict."""


# Query LLM as a judge
def ask_judge(llm_client: LLMClient, system: str, user_message: str, verdict_model: type[T]) -> T:
    try:
        raw = llm_client.chat([{"role": "user", "content": user_message}], system=system, temperature=0.0, prefill="{")
        logger.info("raw LLM judge output: %s", raw)
    except Exception as exc:
        raise JudgeVerificationError(f"Judge LLM call failed: {exc}") from exc

    text = _CODE_FENCE_RE.sub("", raw).strip()
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return verdict_model.model_validate(obj)
    except (ValidationError, json.JSONDecodeError) as exc:
        logger.warning("ask_judge: could not parse verdict from response: %r", raw)
        raise JudgeVerificationError(f"could not parse judge verdict: {exc}") from exc
