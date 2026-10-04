"""
Sparse BM25 retrieval, built per patient-filtered subset at query time.

Isolation happens at index-construction time, not via post-filtering: the
in-memory corpus is grouped by patient_id once at startup, and a fresh
BM25Okapi index is built over just the requested patient's chunks for each
query. At demo, scale this is cheap and gives the same guarantee, in spirit,
as the dense retriever's chroma `where` filter - patient_id filtering happens
before/during retrieval, not as an LLM instruction.

Because that filtering is structural rather than a filter argument, the
isolation guardrail has to be consulted here explicitly: when it is disabled
the search falls back to the flat corpus covering every patient.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from rank_bm25 import BM25Okapi

from app.guardrails.retrieval import PatientIsolationGuardrail
from app.rag.types import Chunk, ScoredChunk


class SparseRetriever(Protocol):
    def search(self, query: str, patient_id: str, top_k: int) -> list[ScoredChunk]: ...


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class Bm25Retriever:
    def __init__(self, chunks: list[Chunk], isolation_guardrail: PatientIsolationGuardrail | None = None) -> None:

        self._all_chunks: list[Chunk] = list(chunks)
        self._by_patient: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            self._by_patient[chunk.patient_id].append(chunk)

        self._isolation_guardrail = isolation_guardrail or PatientIsolationGuardrail()

    def search(self, query: str, patient_id: str, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        
        # With isolation off, every patient's chunks become eligible (the sparse-side equivalent of dropping the chroma `where` clause).
        if self._isolation_guardrail.enabled:
            patient_chunks = self._by_patient.get(patient_id, [])
        else:
            patient_chunks = self._all_chunks
        if not patient_chunks:
            return []

        # Use Okapi BM25 for scoring the chunks
        tokenized_corpus = [_tokenize(c.text) for c in patient_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(query))

        # Keep top k chunks
        scored = [ScoredChunk(chunk=c, score=float(s)) for c, s in zip(patient_chunks, scores)]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]
