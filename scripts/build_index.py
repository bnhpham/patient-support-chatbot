"""
Chunk data/processed/records.jsonl, embed, and build the chroma + BM25 indexes.

Usage:
    python scripts/build_index.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, ChunkingConfig, get_app_config, load_rag_config  # noqa: E402
from app.rag.chunking import Chunker, RecursiveCharacterChunkerAdapter, SemanticChunkerAdapter  # noqa: E402
from app.rag.dense_retriever import DEFAULT_COLLECTION_NAME, open_chroma_collection  # noqa: E402

# ChromaDB rejects a single add() larger than its storage backend allows
# (~5461 records, derived from SQLite's bound-variable limit).
# Adding in batches keeps every call well under that ceiling regardless of corpus size.
CHROMA_ADD_BATCH_SIZE = 1000


# Use either SemanticChunker or RecursiveCharacterChunker
def _build_chunker(chunking: ChunkingConfig, app_config: AppConfig) -> Chunker:

    if chunking.strategy == "semantic":
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=app_config.dense_embedding_model)
        return SemanticChunkerAdapter(embeddings, chunking.semantic_threshold)

    if chunking.strategy == "recursive":
        return RecursiveCharacterChunkerAdapter(chunk_size=chunking.chunk_size,
                                                chunk_overlap=chunking.chunk_overlap)

    raise SystemExit(f"Unknown chunking.strategy {chunking.strategy!r} - expected 'semantic' or 'recursive'.")


def main() -> None:

    # Congig
    app_config = get_app_config()
    rag_config = load_rag_config(app_config.rag_config_path)

    records_path = app_config.processed_dir / "records.jsonl"
    if not records_path.exists():
        raise SystemExit(f"{records_path} not found - run scripts/prepare_medsynth.py first.")

    # Chunker, selected by chunking.strategy in config/rag.yaml
    chunker = _build_chunker(rag_config.chunking, app_config)

    # Chunk MedSynth records
    all_chunks = []
    with open(records_path, "r", encoding="utf-8") as f:
        
        for line in tqdm(f, desc="Chunking"):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            all_chunks.extend(
                chunker.chunk(
                    row["text"],
                    patient_id=row["patient_id"],
                    source_record_id=row["source_record_id"],
                    section=row["section"],
                )
            )

    print(f"Produced {len(all_chunks)} chunks from {records_path}")

    # Build ChromaDB vector database for dense/semantic retrieval
    indexes_dir = app_config.indexes_dir
    chroma_dir = indexes_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    collection = open_chroma_collection(chroma_dir, app_config.dense_embedding_model, DEFAULT_COLLECTION_NAME, reset=True)

    for start in tqdm(range(0, len(all_chunks), CHROMA_ADD_BATCH_SIZE), desc="Build chroma collection"):

        batch = all_chunks[start : start + CHROMA_ADD_BATCH_SIZE]

        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "patient_id": c.patient_id,
                    "source_record_id": c.source_record_id,
                    "section": c.section,
                }
                for c in batch
            ],
        )

    # Store chunks again for later keyword/lexical retrieval via BM25
    bm25_dir = indexes_dir / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)

    with open(bm25_dir / "chunks.jsonl", "w", encoding="utf-8") as f:

        for c in tqdm(all_chunks, desc="Save chunks for BM25 retrieval"):
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "patient_id": c.patient_id,
                        "source_record_id": c.source_record_id,
                        "section": c.section,
                        "text": c.text,
                    }
                )
                + "\n"
            )

    # Documentation
    manifest = {
        "chunk_count": len(all_chunks),
        "patient_count": len({c.patient_id for c in all_chunks}),
        "embedding_model": app_config.dense_embedding_model,
        "chunking_strategy": rag_config.chunking.strategy,
        "semantic_threshold": rag_config.chunking.semantic_threshold,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(indexes_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote chroma collection + BM25 chunks + manifest to {indexes_dir}")


if __name__ == "__main__":
    main()
