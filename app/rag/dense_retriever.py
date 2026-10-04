"""
Dense vector retrieval over a ChromaDB collection, patient-filtered.

The patient_id filter is applied via chroma's `where` clause 
using app.guardrails.retrieval.PatientIsolationGuardrail.where_clause() 
(so before scoring/ranking, not as a post-hoc step).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.guardrails.retrieval import PatientIsolationGuardrail
from app.rag.types import Chunk, ScoredChunk

DEFAULT_COLLECTION_NAME = "patient_chunks"


def open_chroma_collection(persist_directory: Path | str, embedding_model_name: str, 
                           collection_name: str = DEFAULT_COLLECTION_NAME, reset: bool = False):
    """
    Open (or create) the persistent chroma collection used both to build
    the index (scripts/build_index.py) and to query it at runtime, so both
    sides always use the same embedding function.

    `reset=True` drops any existing collection first. Only the build script
    passes it: chunk ids are derived from the chunking config, so rebuilding
    after a config change would otherwise fail on duplicate ids or silently
    leave stale chunks from the previous configuration in the index.
    """
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(persist_directory))

    if reset:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            # Nothing to drop on a first build.
            pass

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)

    return client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)


class DenseRetriever(Protocol):
    def search(self, query: str, patient_id: str, top_k: int) -> list[ScoredChunk]: ...


class ChromaDenseRetriever:
    def __init__(self, collection, isolation_guardrail: PatientIsolationGuardrail) -> None:
        self._collection = collection
        self._isolation_guardrail = isolation_guardrail

    def search(self, query: str, patient_id: str, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            return []

        # Search for relevant chunks
        result = self._collection.query(query_texts=[query], n_results=top_k, where=self._isolation_guardrail.where_clause(patient_id))

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        # ChromaDB returns a distance (lower = more similar).
        # Convert that distance to a similarity score, so every retriever shares a "higher is better" convention 
        # for hybrid_retriever's score merging.
        scored: list[ScoredChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):

            chunk = Chunk(
                chunk_id=chunk_id,
                patient_id=metadata.get("patient_id", ""),
                source_record_id=metadata.get("source_record_id", ""),
                section=metadata.get("section", ""),
                text=text,
            )

            score = 1.0 / (1.0 + distance)
            scored.append(ScoredChunk(chunk=chunk, score=score))

        return scored
