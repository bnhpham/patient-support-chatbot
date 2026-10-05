"""
Shared support module for the poisoned-RAG scenario scripts in tests/attacks/.

These scripts are NOT pytest. Each one builds the REAL FastAPI app in-process
(app.main.create_app with a script-controlled RagConfig override), wrapped in
fastapi.testclient.TestClient - real Chroma/BM25 index, real patient data,
real Claude calls, no separate uvicorn process needed. That is what lets a
script set retrieval.candidate_top_k and a single guardrails.* flag "in the
script": config/rag.yaml is only read once, at app-creation time, so building
the app fresh with an overridden RagConfig is what makes the override take
effect, and it also means each run automatically picks up whatever is
currently poisoned into the persisted index (no backend restart required).

There is no LLM judge and no pass/fail verdict here. Eeach run just saves 
full conversation transcripts to tests/attacks/results/ for manual review.

    terminal:  python tests/attacks/overdose.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Callable  # noqa: E402

from app.config import RagConfig, get_app_config, load_rag_config  # noqa: E402
from app.guardrails.input import InputGuardrailBlocked  # noqa: E402
from app.guardrails.jailbreak import DEFAULT_BLACKLIST_RESPONSE, DEFAULT_BLOCKED_RESPONSE  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.dense_retriever import DEFAULT_COLLECTION_NAME, open_chroma_collection  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from scripts.poison_index import CHUNK_ID, PAYLOADS, _patient_id_for, _rewrite_bm25  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PATIENT_FULL_NAME = "James Whitfield"

# Startup check: Load config and check for API key and data
def preflight() -> None:

    app_config = get_app_config()
    if not app_config.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set (.env) - these scripts make real Claude calls.")

    patients_path = app_config.processed_dir / "patients.jsonl"
    if not patients_path.exists():
        raise SystemExit(f"{patients_path} not found - run scripts/prepare_medsynth.py first.")

    chroma_dir = app_config.indexes_dir / "chroma"
    if not chroma_dir.exists():
        raise SystemExit(f"{chroma_dir} not found - run scripts/build_index.py first.")

    _patient_id_for(PATIENT_FULL_NAME, app_config)


# Plant the scenario's poisoned chunk into James Whitfield's record. Mirrors scripts/poison_index.py's main().
def poison(payload_key: str) -> None:

    # Config
    app_config = get_app_config()
    chroma_dir = app_config.indexes_dir / "chroma"
    bm25_path = app_config.indexes_dir / "bm25" / "chunks.jsonl"

    collection = open_chroma_collection(chroma_dir, app_config.dense_embedding_model, DEFAULT_COLLECTION_NAME)
    try:
        collection.delete(ids=[CHUNK_ID])
    except Exception:
        pass

    # Build poisoned data for James Whitfield
    patient_id = _patient_id_for(PATIENT_FULL_NAME, app_config)
    text = PAYLOADS[payload_key]
    poison_row = {"chunk_id": CHUNK_ID, "patient_id": patient_id, "source_record_id": "dialogue2", "section": "dialogue", "text": text}

    # Add poisoned data to vector database
    collection.add(ids=[CHUNK_ID], documents=[text], metadatas=[{"patient_id": patient_id, "source_record_id": "dialogue2", "section": "dialogue"}])
    _rewrite_bm25(bm25_path, drop_only=False, poison_row=poison_row)
    print(f"  poisoned {PATIENT_FULL_NAME} with payload {payload_key!r} (chunk_id={CHUNK_ID})")


# Undo poison(): restore James Whitfield's record to its unpoisoned state. Mirrors scripts/poison_index.py --remove
def remove_poison() -> None:

    app_config = get_app_config()
    chroma_dir = app_config.indexes_dir / "chroma"
    bm25_path = app_config.indexes_dir / "bm25" / "chunks.jsonl"

    collection = open_chroma_collection(chroma_dir, app_config.dense_embedding_model, DEFAULT_COLLECTION_NAME)
    try:
        collection.delete(ids=[CHUNK_ID])
    except Exception:
        pass

    _rewrite_bm25(bm25_path, drop_only=True, poison_row=None)
    print(f"  removed poison (chunk_id={CHUNK_ID})")


# Use config/rag.yaml, but override retrieval.candidate_top_k and one guardrails.* flag
def build_rag_config(candidate_top_k: int, guardrail_field: str, guardrail_enabled: bool) -> RagConfig:

    base = load_rag_config(get_app_config().rag_config_path)

    retrieval = base.retrieval.model_copy(update={"candidate_top_k": candidate_top_k})
    guardrails = base.guardrails.model_copy(update={guardrail_field: guardrail_enabled})

    return base.model_copy(update={"retrieval": retrieval, "guardrails": guardrails})


# The real app (real index, real patient data, real Claude client) with only rag_config (and
# optionally input_guardrail, for experimenting with an alternative jailbreak_detection backend
# without touching config/rag.yaml or app/main.py's default wiring) overridden
def build_client(rag_config: RagConfig, input_guardrail=None) -> TestClient:
    return TestClient(create_app(rag_config=rag_config, input_guardrail=input_guardrail))


class PluggableJailbreakGuardrail:
    """Same blacklist state machine as app.guardrails.jailbreak.JailbreakGuardrail (a session's 1st
    flagged message gets DEFAULT_BLOCKED_RESPONSE, a 2nd blacklists the rest of that session), but
    scores messages with an injectable score_fn instead of Guardrails AI's DetectJailbreak - lets
    tests/attacks/jailbreak_*.py scripts try a different classifier as the per-message detector
    without touching the production guardrail.
    """

    def __init__(self, score_fn: Callable[[str], float], threshold: float, blacklist_limit: int = 1) -> None:
        self._score_fn = score_fn
        self._threshold = threshold
        self._blacklist_limit = blacklist_limit
        self._flag_counts: dict[str, int] = {}
        self._blacklisted: set[str] = set()

    def check(self, message: str, session_id: str = "default") -> str:

        if session_id in self._blacklisted:
            raise InputGuardrailBlocked(DEFAULT_BLACKLIST_RESPONSE)

        score = self._score_fn(message)
        if score > self._threshold:
            count = self._flag_counts.get(session_id, 0) + 1
            self._flag_counts[session_id] = count

            if count > self._blacklist_limit:
                self._blacklisted.add(session_id)
                raise InputGuardrailBlocked(DEFAULT_BLACKLIST_RESPONSE)

            raise InputGuardrailBlocked(DEFAULT_BLOCKED_RESPONSE)

        return message


def login(client: TestClient, full_name: str = PATIENT_FULL_NAME) -> str:

    r = client.post("/sessions", json={"full_name": full_name})
    if r.status_code != 200:
        raise SystemExit(f"session creation failed ({r.status_code}): {r.text}")
    
    return r.json()["session_id"]


# Send every turn in order over the same session, so history builds up as in a real conversation
def run_conversation(client: TestClient, session_id: str, turns: list[str]) -> list[dict]:

    exchanges: list[dict] = []
    for i, turn in enumerate(turns):

        r = client.post(f"/sessions/{session_id}/chat", json={"message": turn})
        if r.status_code != 200:
            raise SystemExit(f"chat turn failed ({r.status_code}): {r.text}")

        body = r.json()
        body.pop("sources", None)  # not saved to transcripts - noise for manual review
        exchanges.append({"turn_index": i, "request": {"message": turn}, "response": body})

    return exchanges


def delete_session(client: TestClient, session_id: str) -> None:
    try:
        client.delete(f"/sessions/{session_id}")
    except Exception:
        pass


def save_transcript(*, scenario: str, guardrail_field: str, guardrail_enabled: bool, candidate_top_k: int, payload_key: str, conversations: list[dict], note: str | None = None) -> Path:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state = "on" if guardrail_enabled else "off"
    out_path = RESULTS_DIR / f"{scenario}__{guardrail_field}-{state}__{stamp}.json"

    payload = {"scenario": scenario,
               "run_at": datetime.now(timezone.utc).isoformat(),
               "patient_full_name": PATIENT_FULL_NAME,
               "poison_payload": payload_key,
               "config": {"candidate_top_k": candidate_top_k, "guardrail_field": guardrail_field, "guardrail_enabled": guardrail_enabled},
               "note": note,
               "conversations": conversations}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"  saved -> {out_path}")
    return out_path
