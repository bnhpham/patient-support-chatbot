"""
Session storage abstraction.

Sessions hold the patient_id that every later request is authorized against,
so this is the only place that decides how that state is kept. The baseline
keeps it in process memory; the Protocol exists so a persistent store can
replace it without touching SessionService or anything above it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

from app.errors import SessionNotFoundError


@dataclass
class Session:
    session_id: str
    patient_id: str
    full_name: str
    messages: list[dict] = field(default_factory=list)


class SessionRepository(Protocol):
    def create(self, session: Session) -> None: ...
    def get(self, session_id: str) -> Session | None: ...
    def delete(self, session_id: str) -> None: ...
    def append_message(self, session_id: str, role: str, content: str) -> None: ...


class InMemorySessionRepository:
    """
    Process-local session store. Swappable later for a persistent one.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)

            if session is None:
                raise SessionNotFoundError(session_id)

            session.messages.append({"role": role, "content": content})
