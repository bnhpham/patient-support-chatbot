"""
Tracks an accumulating, decaying per-session risk score fed by the trajectory judge (app/guardrails/trajectory.py).

Replaces the hard strike counter (see _StrikeTracker in jailbreak.oy) with a signal that fades over time for isolated weak hits
but accumulates for a sustained or escalating pattern.

Pure state, no detection logic of its own, in-memory and keyed by session_id.
"""

from __future__ import annotations


class SessionRiskBudget:
    def __init__(self, decay: float = 0.85, soft_threshold: float = 0.5, hard_threshold: float = 0.85) -> None:
        self._scores: dict[str, float] = {}
        self._decay = decay
        self._soft_threshold = soft_threshold
        self._hard_threshold = hard_threshold

    # Decay the existing score, then add the new signal (clamped to 1.0). Returns the updated score.
    def record(self, session_id: str, signal: float) -> float:
        current = self._scores.get(session_id, 0.0) * self._decay
        updated = min(1.0, current + signal)
        self._scores[session_id] = updated
        return updated

    def score(self, session_id: str) -> float:
        return self._scores.get(session_id, 0.0)

    def exceeds_soft(self, session_id: str) -> bool:
        return self.score(session_id) >= self._soft_threshold

    def exceeds_hard(self, session_id: str) -> bool:
        return self.score(session_id) >= self._hard_threshold
