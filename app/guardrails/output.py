"""
Output guardrail seam.

Deliberately a no-op in the baseline (see app/guardrails/input.py for the rationale).
"""

from __future__ import annotations

from typing import Protocol


class OutputGuardrail(Protocol):
    def check(self, response: str) -> str: ...


class NoOpOutputGuardrail:
    def check(self, response: str) -> str:
        return response
