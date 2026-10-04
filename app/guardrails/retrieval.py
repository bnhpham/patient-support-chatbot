"""
Patient-level data isolation - the one real guardrail in the baseline.

Every retrieval of medical data must be scoped to session.patient_id.
This guardrail is applied twice, deliberately redundantly:

1. where_clause() produces the metadata filter that dense_retriever.py and
   bm25_retriever.py apply *before/during* retrieval, so chunks belonging to
   other patients are never scored or ranked in the first place.
2. enforce() is a post-hoc assertion pass over whatever chunks came back,
   dropping (and logging) anything that doesn't match - defense in depth
   against a retriever implementation bug, not merely an instruction to the
   LLM.

Both can be switched off together via `guardrails.patient_isolation: false` in
config/rag.yaml, so the leak this guardrail prevents can be demonstrated
side by side with the guarded behaviour. Disabling it is an experiment mode,
never a production option - a disabled run logs a warning on every call so it
cannot be mistaken for normal operation.
"""

from __future__ import annotations

import logging

from app.rag.types import ScoredChunk

logger = logging.getLogger("app.guardrails.retrieval")


class PatientIsolationGuardrail:
    def __init__(self, enabled: bool = True) -> None:
        # Defaults to on, so any caller that does not know about the config
        # flag (tests, ad-hoc scripts) still gets an isolated retriever.
        self.enabled = enabled

    def where_clause(self, patient_id: str) -> dict | None:
        # chroma treats where=None as "no filter", so the retriever itself
        # needs no branching.
        if not self.enabled:
            return None
        return {"patient_id": patient_id}

    # Check if top k chunks belong to given patient_id. Chunks not belonging to the patient are dropped to prevent PII leakage.
    def enforce(self, chunks: list[ScoredChunk], patient_id: str) -> list[ScoredChunk]:
        if not self.enabled:
            logger.warning("PatientIsolationGuardrail is DISABLED (guardrails.patient_isolation=false) - "
                           "passing through %d chunk(s) for patient_id=%s without checking ownership",
                           len(chunks),
                           patient_id)
            return chunks

        allowed, dropped = [], []

        for sc in chunks:
            if sc.chunk.patient_id == patient_id:
                allowed.append(sc)
            else:
                dropped.append(sc)

        if dropped:
            logger.warning("PatientIsolationGuardrail dropped %d chunk(s) not belonging to patient_id=%s: %s",
                           len(dropped),
                           patient_id,
                           [sc.chunk.chunk_id for sc in dropped])
            
        return allowed
