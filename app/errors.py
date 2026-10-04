"""Domain-level exceptions, mapped centrally to HTTP responses in app/main.py.

Services and repositories raise these instead of HTTPException so the
domain layer stays independent of FastAPI.
"""

from __future__ import annotations


class DomainError(Exception):
    """Base class for all domain errors."""


class PatientNotFoundError(DomainError):
    """No patient matches the given full name."""


class AmbiguousPatientNameError(DomainError):
    """More than one patient matches the given full name."""


class SessionNotFoundError(DomainError):
    """No session exists for the given session_id."""


class InvalidRequestError(DomainError):
    """The request is structurally valid but semantically invalid."""
