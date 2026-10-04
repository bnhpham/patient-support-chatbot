"""
Input-side guardrail that checks the full conversation TRAJECTORY rather than any single message to catch multi-turn/slow-boil
manipulation attempts that a per-message classifier (see jailbreak.py) would miss as no individual turn looks malicious on its own.

Runs on a throttle: every "every_n_turns" turns or immediately once the session's SessionRiskBudget score crosses the soft threshold. 
A flagged verdict feeds the risk budget. The session is blocked once the budget crosses the hard threshold.

Toggled by guardrails.trajectory_analysis in config/rag.yaml.

Fail-closed:
A judge-call/parse failure blocks the current user message rather than silently passing it through.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from app.guardrails.input import InputGuardrailBlocked
from app.guardrails.judge_utils import JudgeVerificationError, ask_judge
from app.guardrails.risk_budget import SessionRiskBudget
from app.llm.claude_client import LLMClient
from app.prompts.utils import load_prompt

logger = logging.getLogger("app.guardrails.trajectory")

JUDGE_SYSTEM_PROMPT = load_prompt("trajectory_judge_system.txt")

DEFAULT_UNVERIFIABLE_RESPONSE = ("I'm not able to continue this conversation right now. "
                                 "Please rephrase your question or contact your care team directly.")
DEFAULT_FLAGGED_RESPONSE = ("Please contact your care team directly.")


class TrajectoryVerdict(BaseModel):
    flagged: bool
    score: float
    reason: str


class TrajectoryGuardrail:
    def __init__(self, llm_client: LLMClient, risk_budget: SessionRiskBudget, every_n_turns: int = 3,
                 unverifiable_response: str = DEFAULT_UNVERIFIABLE_RESPONSE,
                 flagged_response: str = DEFAULT_FLAGGED_RESPONSE) -> None:

        self._llm_client = llm_client
        self._risk_budget = risk_budget
        self._every_n_turns = every_n_turns
        self._unverifiable_response = unverifiable_response
        self._flagged_response = flagged_response
        self._turn_counts: dict[str, int] = {}

    # message is the already-cleaned current turn. "history" holds prior turns only (current turn not yet appended).
    def check(self, message: str, session_id: str, history: list[dict]) -> None:

        # Return immediately a fixed answer until the session ends as this converation's malignancy score has exceeded the hard threshold
        if self._risk_budget.exceeds_hard(session_id):
            raise InputGuardrailBlocked(self._flagged_response)

        turn = self._turn_counts.get(session_id, 0) + 1
        self._turn_counts[session_id] = turn

        # Skip trajectory analysis for other turns than "every_n_turns" or if soft threshold is not crossed
        due_scheduled = turn % self._every_n_turns == 0
        due_risk = self._risk_budget.exceeds_soft(session_id)
        if not (due_scheduled or due_risk):
            return

        transcript = _format_transcript(history, message)

        # LLM-as-a-Judge to multi-turn manipulation
        try:
            verdict = ask_judge(self._llm_client, JUDGE_SYSTEM_PROMPT, transcript, TrajectoryVerdict)
        except JudgeVerificationError:
            logger.warning("trajectory judge unavailable/unparseable for session %s - failing closed", session_id)
            raise InputGuardrailBlocked(self._unverifiable_response) from None

        new_score = self._risk_budget.record(session_id, verdict.score)
        logger.info("trajectory_check: session=%s turn=%d score=%.2f flagged=%s reason=%s",
                   session_id, turn, new_score, verdict.flagged, verdict.reason)

        # Return a fixed answer until the session ends as the conversation might contain a multi-turn/slow-boil manipulation
        if verdict.flagged and self._risk_budget.exceeds_hard(session_id):
            raise InputGuardrailBlocked(self._flagged_response)


def _format_transcript(history: list[dict], current_message: str) -> str:
    lines = [f"{turn['role']}: {turn['content']}" for turn in history]
    lines.append(f"user: {current_message}")
    return "\n".join(lines)
