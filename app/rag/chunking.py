"""
Chunking strategies for MedSynth records.

Used offline by scripts/build_index.py - not on the request (chat) path.
"""

from __future__ import annotations

from typing import Protocol

from app.rag.types import Chunk


class Chunker(Protocol):
    def chunk(
        self, text: str, *, patient_id: str, source_record_id: str, section: str
    ) -> list[Chunk]: ...


class SemanticChunkerAdapter:
    """
    Wraps langchain_experimental's SemanticChunker.

    "semantic_threshold" (config: chunking.semantic_threshold, conceptually "Semantic 80") 
    is passed as breakpoint_threshold_amount with breakpoint_threshold_type="percentile": 
    
    Within a document, consecutive sentences are embedded and the distance between each 
    adjacent pair is computed. A new chunk boundary is inserted only where that distance
    is at or above the given percentile of all distances observed in that document
    - e.g. 80 means only the most surprising 20% of sentence-to-sentenc etopic shifts
    become chunk boundaries. Higher values produce fewer, larger chunks.
    """

    def __init__(self, embeddings, semantic_threshold: int) -> None:
        from langchain_experimental.text_splitter import SemanticChunker

        self._splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=semantic_threshold,
        )

    def chunk(self, text: str, *, patient_id: str, source_record_id: str, section: str) -> list[Chunk]:
        if not text or not text.strip():
            return []
        
        pieces = self._splitter.split_text(text)
        
        return [
            Chunk(
                chunk_id=f"{source_record_id}:{section}:{i}",
                patient_id=patient_id,
                source_record_id=source_record_id,
                section=section,
                text=piece,
            )
            for i, piece in enumerate(pieces)
        ]


class RecursiveCharacterChunkerAdapter:
    """
    Fixed-size fallback chunker - fast, no embedding model required.

    Used by tests and available as a documented fallback if chunking.strategy is ever set to something other than "semantic".
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, text: str, *, patient_id: str, source_record_id: str, section: str) -> list[Chunk]:
        if not text or not text.strip():
            return []
        
        pieces = self._splitter.split_text(text)

        return [
            Chunk(
                chunk_id=f"{source_record_id}:{section}:{i}",
                patient_id=patient_id,
                source_record_id=source_record_id,
                section=section,
                text=piece,
            )
            for i, piece in enumerate(pieces)
        ]
