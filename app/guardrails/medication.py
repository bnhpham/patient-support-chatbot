"""
Medication guardrail - a knowledge-base cross-check.

This one validates the *facts* in the retrieved patient_context:
any medication mentioned there is looked up in a trusted external reference (data/drugs/drugs.yaml),
and the reference - plus any deterministic dosage conflict - is added to the user message.
The assistant can then notice when the record disagrees with the reference, warn the patient, and advise contacting the doctor.

Toggled by guardrails.fact_check in config/rag.yaml. When off, the guardrail is never consulted and the prompt is unchanged.

Detection is two-pass: a transparent alias table built from drugs.yaml (primary), then an optional drug-named-entity-recognition
pass for brand names and misspellings the table misses. The NER import is guarded, so the app and the tests run fine without
that package installed.
"""

from __future__ import annotations

import logging
import re

import yaml

logger = logging.getLogger("app.guardrails.medication")

# Matches "3600 mg", "3,600mg", "800 MG" - the number is captured without commas.
_MG_RE = re.compile(r"(\d[\d,]*)\s*mg\b", re.IGNORECASE)


class MedicationGuardrail:
    def __init__(self, drugs_path) -> None:

        # Load drugs from data/drugs/drugs.yaml
        with open(drugs_path, "r", encoding="utf-8") as f:
            self._drugs: dict[str, dict] = yaml.safe_load(f) or {}

        # Maps pharmaceutical terms to a canonical drug name
        # Example:
        #{
        #    "ibuprofen": "ibuprofen",
        #    "NSAID": "ibuprofen",
        #    "Motrin": "ibuprofen",
        #}
        self._alias_to_canonical: dict[str, str] = {}

        for canonical, entry in self._drugs.items():
            self._alias_to_canonical[canonical.lower()] = canonical

            # Add terms listed in relations: [...] (see drugs.yaml) to dict
            for alias in (entry or {}).get("relations", []) or []:
                self._alias_to_canonical[str(alias).lower()] = canonical

    # Detect Canonical drug names mentioned in `text` (alias table + optional NER).
    def detect(self, text: str) -> set[str]:

        low = text.lower()

        # Based dict based on our custom drug database (drugs.yaml)
        found = {canonical for alias, canonical in self._alias_to_canonical.items() if re.search(rf"\b{re.escape(alias)}\b", low)}

        # Add drug names found by the Drug Named Entity Recognition Package
        found |= self._detect_ner(text)
        
        return found

    # Drug Named Entity Recognition Package
    def _detect_ner(self, text: str) -> set[str]:

        from drug_named_entity_recognition import find_drugs

        try:
            matches = find_drugs(text.split(), is_ignore_case=True)
        # Never let the NER pass break a chat turn
        except Exception as exc:
            logger.warning("drug NER pass failed, ignoring: %s", exc)
            return set()

        found: set[str] = set()

        for match in matches:

            entry = match[0] if isinstance(match, tuple) else match
            name = (entry.get("name") if isinstance(entry, dict) else str(entry)) or ""

            canonical = self._alias_to_canonical.get(name.lower())
            if canonical:
                found.add(canonical)

        return found

    def find_conflicts(self, text: str, drugs: set[str]) -> list[str]:
        """
        Deterministic dosage check: flag any mg value above a drug's ceiling.

        The ceiling is rx_max_daily_mg (the highest legitimate figure), falling
        back to otc_max_daily_mg when Rx is null. Conservative on purpose: a
        3,600 mg claim flags, a normal 1,200 mg dose does not. It does not try
        to distinguish single-dose from daily totals - a documented limitation.
        """

        values = [int(m.group(1).replace(",", "")) for m in _MG_RE.finditer(text)]
        if not values:
            return []

        conflicts: list[str] = []
        for canonical in sorted(drugs):

            entry = self._drugs.get(canonical) or {}

            ceiling = entry.get("rx_max_daily_mg") or entry.get("otc_max_daily_mg")
            if not ceiling:
                continue

            for value in values:
                if value > ceiling:
                    conflicts.append(f"CONFLICT: the record mentions {value} mg, but the trusted maximum for "
                                     f"{canonical} is {ceiling} mg/day ({entry.get('source', 'reference')}).")
                    
        return conflicts
    
    def reference_block(self, chunks) -> str | None:
        """
        Reference + conflict block for the drugs in "chunks", or None if none.
        "chunks" is the reranked list of ScoredChunk. Returning None leaves the prompt untouched.
        Function is used by pipline.py
        """

        text = "\n".join(sc.chunk.text for sc in chunks)
        return self.reference_block_from_text(text)

    def reference_block_from_text(self, text: str) -> str | None:

        # Extract drugs from chunks
        drugs = self.detect(text)
        if not drugs:
            return None

        lines = ["AUTHORITATIVE MEDICATION REFERENCE (trusted):",
                 "Compare the patient context above against the reference below. If any dosage or "
                 "instruction in the context disagrees with this reference, do NOT repeat the context's "
                 "figure - tell the patient there is a discrepancy and advise them to contact their doctor.",
                 ""]

        # Look up for maximum dosage for each drug
        for canonical in sorted(drugs):
            entry = self._drugs.get(canonical) or {}
            lines.append(f"- {canonical}: OTC max {entry.get('otc_max_daily_mg', 'n/a')} mg/day, "
                         f"Rx max {entry.get('rx_max_daily_mg', 'n/a')} mg/day, "
                         f"single dose max {entry.get('otc_max_single_mg', 'n/a')} mg "
                         f"(source: {entry.get('source', 'reference')}).")

        # Find conflicts between data from loaded chunks and that from our custom drug database
        conflicts = self.find_conflicts(text, drugs)
        if conflicts:
            lines.append("")
            lines.append("DETECTED CONFLICTS:")
            lines.extend(f"- {c}" for c in conflicts)
            logger.info("medication_guardrail: %d conflict(s) detected: %s", len(conflicts), conflicts)

        return "\n".join(lines)
