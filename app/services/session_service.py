"""
Session lifecycle: patient verification by full_name, in-memory session state.

patient_id is resolved once, server-side, at session-creation time and lives
only inside the Session object from then on. Nothing in the API layer ever
reads a patient_id from a client request - every later operation (chat,
appointments) must go through get_session() to recover it.

Session, SessionRepository and InMemorySessionRepository live in
app/repositories/session_repository.py, alongside the patient and appointment
repositories.
"""

from __future__ import annotations

import uuid

from app.errors import PatientNotFoundError, SessionNotFoundError
from app.repositories.session_repository import Session, SessionRepository
from app.services.patient_service import PatientService


class SessionService:
    def __init__(self, repo: SessionRepository, patient_service: PatientService) -> None:

        # Save all sessions in SessionRepository "self._repo"
        self._repo = repo
        self._patient_service = patient_service

    def create_session(self, full_name: str) -> Session:
        # Raises PatientNotFoundError/AmbiguousPatientNameError handled by the router.
        # find_by_full_name never returns a random match on ambiguity.
        patient = self._patient_service.find_by_full_name(full_name)
        if patient is None:
            raise PatientNotFoundError(full_name)

        session = Session(session_id=str(uuid.uuid4()), patient_id=patient.patient_id, full_name=patient.full_name)
        self._repo.create(session)
        return session

    def get_session(self, session_id: str) -> Session:
        session = self._repo.get(session_id)

        if session is None:
            raise SessionNotFoundError(session_id)

        return session

    def delete_session(self, session_id: str) -> None:
        self._repo.delete(session_id)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self._repo.append_message(session_id, role, content)
