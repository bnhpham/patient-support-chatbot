"""
Output-side policy/ethics check:
asks an LLM judge whether the drafted answer complies with a medical-conduct rubric (see policy_judge_system.txt for the exact rubric).

Runs as a node inside RagPipeline's graph (see app/rag/pipeline.py).
On a flag, the pipeline regenerates the answer with the violation fed back (see validate_output in pipeline.py).

Toggled by guardrails.policy_check in config/rag.yaml.

Fail-closed:
If JudgeVerificationError occurs (judge call fails or unparseable output), the answer is treated as flagged rather than passed through.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from app.guardrails.judge_utils import JudgeVerificationError, ask_judge
from app.llm.claude_client import LLMClient
from app.prompts.utils import load_prompt

logger = logging.getLogger("app.guardrails.policy")

JUDGE_SYSTEM_PROMPT = load_prompt("policy_judge_system.txt")

DEFAULT_UNVERIFIED_REASON = "policy check could not be completed (judge unavailable)"


class PolicyVerdict(BaseModel):
    passed: bool
    violated_rule: str | None = None
    reason: str | None = None


class PolicyGuardrail:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    # Returns a reason string if the answer should be regenerated, None if it's clean.
    def check(self, answer: str) -> str | None:

        # LLM-as-a-Judge to detect policy violation
        try:
            verdict = ask_judge(self._llm_client, JUDGE_SYSTEM_PROMPT, answer, PolicyVerdict)
        except JudgeVerificationError:
            logger.warning("policy judge unavailable/unparseable - failing closed")
            return DEFAULT_UNVERIFIED_REASON

        if verdict.passed:
            return None

        logger.info("policy_check: flagged - rule=%s reason=%s", verdict.violated_rule, verdict.reason)
        return f"policy violation (rule {verdict.violated_rule}): {verdict.reason}"
