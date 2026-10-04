"""
Output-side hallucination/groundedness check:
asks an LLM judge whether each factual claim in the drafted answer is supported by the retrieved context.

Runs as a node inside the RAG pipeline's graph (see app/rag/pipeline.py), not as an OutputGuardrail, because it needs direct
access to the retrieved chunks that produced the answer, not just the final text in isolation.
On a flag, the pipeline regenerates the answer with the violation fed back (see validate_output_node in pipeline.py).

Toggled by guardrails.hallucination_check in config/rag.yaml.

Fail-closed:
If JudgeVerificationError occurs (judge call fails or unparseable output), the answer is treated as flagged rather than passed through.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from app.guardrails.judge_utils import JudgeVerificationError, ask_judge
from app.llm.claude_client import LLMClient
from app.prompts.utils import load_prompt
from app.rag.types import ScoredChunk

logger = logging.getLogger("app.guardrails.hallucination")

JUDGE_SYSTEM_PROMPT = load_prompt("hallucination_judge_system.txt")

DEFAULT_UNVERIFIED_REASON = "hallucination check could not be completed (judge unavailable)"


class ClaimAssessment(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "contradicted"]
    severity: Literal["material", "minor"]


class HallucinationVerdict(BaseModel):
    claims: list[ClaimAssessment]


class HallucinationGuardrail:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    # Returns a reason string if the answer should be regenerated, None if it's clean.
    def check(self, answer: str, chunks: list[ScoredChunk]) -> str | None:

        context_text = _format_chunks(chunks)
        user_message = f"<context>\n{context_text}\n</context>\n\n<answer>\n{answer}\n</answer>"

        # LLM-as-a-Judge to detect hallucination
        try:
            verdict = ask_judge(self._llm_client, JUDGE_SYSTEM_PROMPT, user_message, HallucinationVerdict)
        except JudgeVerificationError:
            logger.warning("hallucination judge unavailable/unparseable - failing closed")
            return DEFAULT_UNVERIFIED_REASON

        offending = [c for c in verdict.claims if c.verdict != "supported" and c.severity == "material"]
        if not offending:
            return None

        reasons = "; ".join(f'"{c.claim}" ({c.verdict})' for c in offending)
        logger.info("hallucination_check: flagged claims: %s", reasons)
        return f"unsupported medical claim(s): {reasons}"


def _format_chunks(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return "(no retrieved context)"
    
    return "\n\n".join(f"[{sc.chunk.section} | {sc.chunk.source_record_id}] {sc.chunk.text}" for sc in chunks)
