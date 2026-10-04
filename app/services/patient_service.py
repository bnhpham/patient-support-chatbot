"""Thin lookup wrapper over PatientRepository, used only by SessionService.

Patient data is never exposed through the API directly - this service exists
so that session creation has a single, obvious place to resolve full_name to
a patient without reaching into the repository layer from multiple services.
"""

from __future__ import annotations

from app.repositories.patient_repository import Patient, PatientRepository


class PatientService:
    def __init__(self, repo: PatientRepository) -> None:
        self._repo = repo

    def find_by_full_name(self, full_name: str) -> Patient | None:
        return self._repo.find_by_full_name(full_name)
