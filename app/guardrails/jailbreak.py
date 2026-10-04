"""
Jailbreak guardrail - Guardrails AI's DetectJailbreak validator.

Runs on the raw user message, before it reaches the RAG pipeline. On trigger it does not try to repair or reask.
It raises InputGuardrailBlocked so chat_service returns a fixed response and the LLM is never called.

On top of the per-message DetectJailbreak check, this guardrail can optionally track flags per session_id via _StrikeTracker:
    - blacklist_limit=1 means a session's 1st flagged message gets blocked_response.
    - A 2nd flagged message blacklists that session outright.
    - Every message for the rest of it, flagged or not, gets blacklist_response without even reaching the detector, until the session ends.

This escalation is now OFF by default. It is a per-message-only risk signal and is blind to attacks spread across many turns,
each individually unremarkable.

That gap is covered by TrajectoryGuardrail + SessionRiskBudget, which look at the whole conversation rather than a strike count.
The strike mechanism is kept and remains available via strike_tracking=True for anyone who wants the simpler, stricter behavior back.

Tracking, when enabled, is in-memory, keyed by session_id, and is never
leaned up on session deletion (harmless: session ids are never reused).

Toggled by guardrails.jailbreak_detection in config/rag.yaml. When off, this module is never constructed.
"""

from __future__ import annotations

from guardrails import Guard
from guardrails_ai.detect_jailbreak import DetectJailbreak

from app.guardrails.input import InputGuardrailBlocked

DEFAULT_BLOCKED_RESPONSE = "I can only help with questions about your previous doctor conversation."
DEFAULT_BLACKLIST_RESPONSE = "Multiple jailbreak attempts detected. You have now been blacklisted!"


# Per-session flag count -> permanent blacklist once blacklist_limit is crossed.
class _StrikeTracker:
    def __init__(self, blacklist_limit: int = 1) -> None:
        self._blacklist_limit = blacklist_limit
        self._flag_counts: dict[str, int] = {}
        self._blacklisted: set[str] = set()

    def is_blacklisted(self, session_id: str) -> bool:
        return session_id in self._blacklisted

    # Records a flag for session_id. Returns True if this flag just crossed the blacklist limit.
    def flag(self, session_id: str) -> bool:
        count = self._flag_counts.get(session_id, 0) + 1
        self._flag_counts[session_id] = count

        if count > self._blacklist_limit:
            self._blacklisted.add(session_id)
            return True
        return False


class JailbreakGuardrail:
    def __init__(self, guard: Guard | None = None, threshold: float = 0.9, strike_tracking: bool = False, blacklist_limit: int = 1,
                 blocked_response: str = DEFAULT_BLOCKED_RESPONSE, blacklist_response: str = DEFAULT_BLACKLIST_RESPONSE) -> None:

        self._guard = guard or Guard().use(DetectJailbreak(threshold=threshold, on_fail="noop"))
        self._blocked_response = blocked_response
        self._blacklist_response = blacklist_response
        self._strikes = _StrikeTracker(blacklist_limit) if strike_tracking else None

    # Validate user's input message. session_id tracks repeated flags across a session - see the module docstring.
    def check(self, message: str, session_id: str = "default") -> str:

        if self._strikes is not None and self._strikes.is_blacklisted(session_id):
            raise InputGuardrailBlocked(self._blacklist_response)

        outcome = self._guard.validate(message)

        if not outcome.validation_passed:
            if self._strikes is not None and self._strikes.flag(session_id):
                raise InputGuardrailBlocked(self._blacklist_response)

            raise InputGuardrailBlocked(self._blocked_response)

        return message
