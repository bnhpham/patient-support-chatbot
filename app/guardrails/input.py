"""
Input guardrail seam.

This is deliberately a no-op in the baseline. The goal of this project is to later measure the effect of adding real input guardrails
(e.g., prompt-injection or jailbreak detectors) by comparing this baseline against a guarded version. 
chat_service.py depends on the InputGuardrail protocol, so swapping in a real implementation later requires no changes outside this file.

A guardrail that must block the message entirely (e.g., a detected jailbreak attempt, where the LLM should never see the input)
raises InputGuardrailBlocked. chat_service.py catches it and short-circuits with the fixed response.

check() also receives session_id, for guardrails that need to remember something across a session's turns
(e.g. JailbreakGuardrail's per-session blacklist after repeated flags) rather than judging each message in isolation.
NoOpInputGuardrail ignores it.
"""

from __future__ import annotations

from typing import Protocol


class InputGuardrail(Protocol):
    def check(self, message: str, session_id: str = "") -> str: ...


# Raised by InputGuardrail.check() to short-circuit chat_service.handle_chat before the RAG pipeline runs. 
# "response" is the fixed text to show the user instead.
class InputGuardrailBlocked(Exception):
    def __init__(self, response: str) -> None:
        super().__init__(response)
        self.response = response


class NoOpInputGuardrail:
    def check(self, message: str, session_id: str = "") -> str:
        return message
