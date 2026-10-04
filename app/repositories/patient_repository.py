"""
Patient lookup abstraction.

Patients are demo data prepared offline by scripts/prepare_medsynth.py into
data/processed/patients.jsonl. patient_id is assigned once during that
preprocessing step and is treated as a stable, opaque identifier from here
on - it is never re-derived from full_name at lookup time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.errors import AmbiguousPatientNameError


@dataclass(frozen=True)
class Patient:
    patient_id: str
    full_name: str
    doctor_patient_dialogue: str = ""
    note: str = ""
    diagnosis: str = ""
    source_record_id: str = ""


def _normalize_name(full_name: str) -> str:
    return " ".join(full_name.strip().lower().split())


class PatientRepository(Protocol):
    def find_by_full_name(self, full_name: str) -> Patient | None:
        """
        Return the matching patient, or None if no patient matches.

        Raises AmbiguousPatientNameError if more than one patient matches -
        callers must handle this explicitly rather than silently picking one.
        """
        ...

    def get_by_id(self, patient_id: str) -> Patient | None: ...


class InMemoryPatientRepository:
    """
    Patient repository backed by an in-memory list, usable directly by tests.
    """

    def __init__(self, patients: list[Patient]) -> None:
        self._by_id: dict[str, Patient] = {p.patient_id: p for p in patients}
        self._by_name: dict[str, list[Patient]] = defaultdict(list)
        for p in patients:
            self._by_name[_normalize_name(p.full_name)].append(p)

    def find_by_full_name(self, full_name: str) -> Patient | None:
        matches = self._by_name.get(_normalize_name(full_name), [])
        if not matches:
            return None

        # Reject ambiguity
        if len(matches) > 1:
            raise AmbiguousPatientNameError(f"{len(matches)} patients match full_name={full_name!r}")
        
        return matches[0]

    def get_by_id(self, patient_id: str) -> Patient | None:
        return self._by_id.get(patient_id)


class JsonPatientRepository(InMemoryPatientRepository):
    """
    Loads patients from a JSONL file produced by scripts/prepare_medsynth.py.
    """

    def __init__(self, patients_path: Path | str) -> None:
        patients: list[Patient] = []

        with open(patients_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                patients.append(
                    Patient(
                        patient_id=row["patient_id"],
                        full_name=row["full_name"],
                        doctor_patient_dialogue=row.get("doctor_patient_dialogue", ""),
                        note=row.get("note", ""),
                        diagnosis=row.get("diagnosis", ""),
                        source_record_id=row.get("source_record_id", ""),
                    )
                )
                
        super().__init__(patients)
