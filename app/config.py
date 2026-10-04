"""
Application configuration: environment variables (.env) + config/rag.yaml.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AppConfig(BaseSettings):
    """
    Runtime environment configuration, loaded from process env / .env.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    anthropic_max_tokens: int = 16000

    # NOTE: scripts/build_index.py (index time) and the dense retriever (query time) must use the SAME embedding model.
    # If they differ, ChromaDB returns nearest neighbours in a mismatched vector space, so retrieval silently degrades to noise.
    dense_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    rag_config_path: Path = PROJECT_ROOT / "config" / "rag.yaml"

    data_dir: Path = PROJECT_ROOT / "data"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    indexes_dir: Path = PROJECT_ROOT / "data" / "indexes"
    drugs_path: Path = PROJECT_ROOT / "data" / "drugs" / "drugs.yaml"


class ChunkingConfig(BaseModel):
    strategy: str = "semantic"
    semantic_threshold: int = 80
    # Used only by the "recursive" strategy.
    chunk_size: int = 800
    chunk_overlap: int = 100


class RetrievalConfig(BaseModel):
    strategy: str = "hybrid"
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    candidate_top_k: int = 30


class GenerationConfig(BaseModel):
    orchestration: str = "stuff"
    temperature: float = 0.0
    max_regenerations: int = 2


class GuardrailsConfig(BaseModel):
    patient_isolation: bool = True
    fact_check: bool = False
    jailbreak_detection: bool = False
    jailbreak_threshold: float = 0.725
    jailbreak_strike_tracking: bool = False
    rag_injection_detection: bool = False
    trajectory_analysis: bool = False
    hallucination_check: bool = False
    policy_check: bool = False


# NOTE: The values below are FALLBACKS, applied only when a key is absent from config/rag.yaml.
# That YAML file is the live configuration. It is what the running app and the experiment scripts actually read.
class RagConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()


def load_rag_config(path: Path | str | None = None) -> RagConfig:
    """
    Load config/rag.yaml into a validated RagConfig.
    """
    
    config_path = Path(path) if path is not None else get_app_config().rag_config_path
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RagConfig.model_validate(raw)


@lru_cache
def get_app_config() -> AppConfig:
    return AppConfig()
