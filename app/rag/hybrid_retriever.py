"""
Hybrid retrieval: dense + sparse retriever, min-max normalized and weight-merged.

Score merging, documented explicitly since later guardrail experiments will
want to reason about it:

1. Run dense and sparse search independently for each sub-query, each
   returning up to `candidate_top_k` results, both using a "higher is
   better" score convention.
2. Within EACH retriever's result set, min-max normalize scores to [0, 1]
   (a set of all-equal scores normalizes every item to 1.0).
3. merged_score = dense_weight * norm_dense + sparse_weight * norm_sparse,
   where a chunk missing from one retriever's results contributes 0 for that side.
4. Merge across all sub-queries by chunk_id, keeping the max merged score
   per chunk, sort, then truncate to `candidate_top_k`.

app.guardrails.retrieval.PatientIsolationGuardrail.enforce() runs as a final,
independent check over the merged candidates before they leave this stage.
"""

from __future__ import annotations

import logging

from app.guardrails.retrieval import PatientIsolationGuardrail
from app.rag.bm25_retriever import SparseRetriever
from app.rag.dense_retriever import DenseRetriever
from app.rag.types import ScoredChunk

logger = logging.getLogger("app.rag.hybrid_retriever")


# Min-Max Normalization
def _min_max_normalize(scored: list[ScoredChunk]) -> dict[str, float]:
    if not scored:
        return {}
    
    scores = [sc.score for sc in scored]
    low, high = min(scores), max(scores)

    # Chunks get 1 as scores if all scores are equal
    if high == low:
        return {sc.chunk.chunk_id: 1.0 for sc in scored}
    
    return {sc.chunk.chunk_id: (sc.score - low) / (high - low) for sc in scored}


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, isolation_guardrail: PatientIsolationGuardrail) -> None:
        self._dense = dense
        self._sparse = sparse
        self._isolation_guardrail = isolation_guardrail

    def retrieve(self, sub_queries: list[str], patient_id: str, dense_weight: float, sparse_weight: float, candidate_top_k: int
                 ) -> list[ScoredChunk]:
        
        merged: dict[str, ScoredChunk] = {}

        for sub_query in sub_queries:
            # Find chunks by running dense and sparse retriever independently
            dense_results = self._dense.search(sub_query, patient_id, candidate_top_k)
            sparse_results = self._sparse.search(sub_query, patient_id, candidate_top_k)

            # Min-Max Normalization
            dense_norm = _min_max_normalize(dense_results)
            sparse_norm = _min_max_normalize(sparse_results)

            chunk_by_id = {sc.chunk.chunk_id: sc.chunk for sc in [*dense_results, *sparse_results]}
            all_ids = set(dense_norm) | set(sparse_norm)

            for chunk_id in all_ids:
                # Compute merged score for chunk. If chunk is missing in one retriever's results, assign 0.0 for that side
                merged_score = dense_weight * dense_norm.get(chunk_id, 0.0) + sparse_weight * sparse_norm.get(chunk_id, 0.0)
                candidate = ScoredChunk(chunk=chunk_by_id[chunk_id], score=merged_score)

                # Save chunk and its score. If chunk was contained in a previous subquery and new calculated score is bigger --> update chunk in dict
                existing = merged.get(chunk_id)
                if existing is None or candidate.score > existing.score:
                    merged[chunk_id] = candidate

        # Keep top k chunks
        candidates = sorted(merged.values(), key=lambda sc: sc.score, reverse=True)[:candidate_top_k]

        # Check if top k chunks belong to given patient_id. Chunks not belonging to the patient are dropped to prevent PII leakage.
        safe_candidates = self._isolation_guardrail.enforce(candidates, patient_id)

        logger.debug("hybrid_retriever: patient_id=%s sub_queries=%s candidates=%s", patient_id, sub_queries, 
                     [(sc.chunk.chunk_id, round(sc.score, 4)) for sc in safe_candidates])
        return safe_candidates
