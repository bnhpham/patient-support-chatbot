"""
FastAPI application factory.

create_app() accepts every collaborator as an optional override, so tests can
inject fakes/fixtures without touching real data, the filesystem, or the
Claude API. With no overrides it wires up the real demo stack, falling back
to empty in-memory repositories/indexes if scripts/*.py haven't been run yet.
The app still starts, but it just has no patient data until data is prepared.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import sessions as sessions_api
from app.config import AppConfig, RagConfig, get_app_config, load_rag_config
from app.errors import AmbiguousPatientNameError, DomainError, InvalidRequestError, PatientNotFoundError, SessionNotFoundError
from app.guardrails.hallucination import HallucinationGuardrail
from app.guardrails.input import InputGuardrail, NoOpInputGuardrail
from app.guardrails.jailbreak import JailbreakGuardrail
from app.guardrails.medication import MedicationGuardrail
from app.guardrails.output import NoOpOutputGuardrail, OutputGuardrail
from app.guardrails.policy import PolicyGuardrail
from app.guardrails.rag_injection import RagInjectionGuardrail
from app.guardrails.retrieval import PatientIsolationGuardrail
from app.guardrails.risk_budget import SessionRiskBudget
from app.guardrails.trajectory import TrajectoryGuardrail
from app.llm.claude_client import ClaudeClient, LLMClient
from app.rag.bm25_retriever import Bm25Retriever, SparseRetriever
from app.rag.dense_retriever import ChromaDenseRetriever, DenseRetriever, open_chroma_collection
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.pipeline import RagPipeline
from app.rag.types import Chunk
from app.repositories.patient_repository import InMemoryPatientRepository, JsonPatientRepository, PatientRepository
from app.repositories.session_repository import InMemorySessionRepository, SessionRepository
from app.services.chat_service import ChatService
from app.services.patient_service import PatientService
from app.services.session_service import SessionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

FRONTEND_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app(
    *,
    app_config: AppConfig | None = None,
    rag_config: RagConfig | None = None,
    patient_repository: PatientRepository | None = None,
    session_repository: SessionRepository | None = None,
    llm_client: LLMClient | None = None,
    dense_retriever: DenseRetriever | None = None,
    sparse_retriever: SparseRetriever | None = None,
    input_guardrail: InputGuardrail | None = None,
    output_guardrail: OutputGuardrail | None = None,
    isolation_guardrail: PatientIsolationGuardrail | None = None,
    medication_guardrail: MedicationGuardrail | None = None,
    rag_injection_guardrail: RagInjectionGuardrail | None = None,
    trajectory_guardrail: TrajectoryGuardrail | None = None,
    hallucination_guardrail: HallucinationGuardrail | None = None,
    policy_guardrail: PolicyGuardrail | None = None,
) -> FastAPI:

    # Config
    app_config = app_config or get_app_config()
    rag_config = rag_config or load_rag_config(app_config.rag_config_path)

    # Repositories
    patient_repository = patient_repository or _default_patient_repository(app_config)
    session_repository = session_repository or InMemorySessionRepository()

    # LLM
    llm_client = llm_client or ClaudeClient(api_key=app_config.anthropic_api_key, 
                                            model=app_config.anthropic_model,
                                            max_tokens=app_config.anthropic_max_tokens)

    # One shared instance: the dense retriever, the sparse retriever and the hybrid merge must all see the same on/off state, 
    # or disabling isolation only half-works.
    isolation_guardrail = isolation_guardrail or PatientIsolationGuardrail(enabled=rag_config.guardrails.patient_isolation)
    if not isolation_guardrail.enabled:
        logger.warning("PATIENT ISOLATION IS DISABLED (guardrails.patient_isolation=false in %s). "
                       "Retrieval will return chunks belonging to any patient. Experiment mode only.",
                       app_config.rag_config_path)

    # Services
    patient_service = PatientService(patient_repository)
    session_service = SessionService(session_repository, patient_service)

    # RAG pipline & guardrails within the pipeline
    dense_retriever = dense_retriever or _default_dense_retriever(app_config, isolation_guardrail)
    sparse_retriever = sparse_retriever or _default_sparse_retriever(app_config, isolation_guardrail)
    hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever, isolation_guardrail)
    medication_guardrail = medication_guardrail or MedicationGuardrail(app_config.drugs_path)
    rag_injection_guardrail = rag_injection_guardrail or (_default_rag_injection_guardrail(rag_config.guardrails.rag_injection_threshold) if rag_config.guardrails.rag_injection_detection else None)
    hallucination_guardrail = hallucination_guardrail or (HallucinationGuardrail(llm_client=llm_client) if rag_config.guardrails.hallucination_check else None)
    policy_guardrail = policy_guardrail or (PolicyGuardrail(llm_client=llm_client) if rag_config.guardrails.policy_check else None)
    rag_pipeline = RagPipeline(hybrid_retriever, llm_client, rag_config,
                               medication_guardrail, rag_injection_guardrail,
                               hallucination_guardrail, policy_guardrail)

    # Additional guardrails
    input_guardrail = input_guardrail or (JailbreakGuardrail(threshold=rag_config.guardrails.jailbreak_threshold,
                                                             strike_tracking=rag_config.guardrails.jailbreak_strike_tracking,
                                                             blacklist_limit=rag_config.guardrails.jailbreak_blacklist_limit)
                                          if rag_config.guardrails.jailbreak_detection else NoOpInputGuardrail())
    output_guardrail = output_guardrail or NoOpOutputGuardrail()

    # SessionRiskBudget backs only the trajectory guardrail (see risk_budget.py docstring)
    # (not shared with JailbreakGuardrail's own (now-optional) strike tracking).
    risk_budget = SessionRiskBudget()
    trajectory_guardrail = trajectory_guardrail or (TrajectoryGuardrail(llm_client=llm_client, risk_budget=risk_budget)
                                                    if rag_config.guardrails.trajectory_analysis else None)

    chat_service = ChatService(session_service, rag_pipeline, input_guardrail, output_guardrail, trajectory_guardrail)

    # Final FastAPI application
    fastapi_app = FastAPI(title="Medbot - A Patient Support Chatbot")
    fastapi_app.add_middleware(CORSMiddleware, allow_origins=FRONTEND_ORIGINS, allow_methods=["*"], allow_headers=["*"])

    fastapi_app.state.session_service = session_service
    fastapi_app.state.chat_service = chat_service

    fastapi_app.include_router(sessions_api.router)

    _register_exception_handlers(fastapi_app)

    return fastapi_app


# Patient repository used when running the app with "uvicorn app.main:create_app --factory"
def _default_patient_repository(app_config: AppConfig) -> PatientRepository:

    patients_path = app_config.processed_dir / "patients.jsonl"
    if patients_path.exists():
        return JsonPatientRepository(patients_path)
    
    logger.warning("No patient data found at %s - run scripts/prepare_medsynth.py. Starting with an empty patient repository.", patients_path)
    return InMemoryPatientRepository([])


# Dense retriever used when running the app with "uvicorn app.main:create_app --factory"
def _default_dense_retriever(app_config: AppConfig, isolation_guardrail: PatientIsolationGuardrail) -> DenseRetriever:

    collection = open_chroma_collection(persist_directory=app_config.indexes_dir / "chroma", embedding_model_name=app_config.dense_embedding_model)
    return ChromaDenseRetriever(collection, isolation_guardrail)


# Sparse retriever used when running the app with "uvicorn app.main:create_app --factory"
def _default_sparse_retriever(app_config: AppConfig, isolation_guardrail: PatientIsolationGuardrail) -> SparseRetriever:
    chunks_path = app_config.indexes_dir / "bm25" / "chunks.jsonl"
    chunks: list[Chunk] = []
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk(**json.loads(line)))
    else:
        logger.warning("No BM25 chunk index found at %s - run scripts/build_index.py. Starting with an empty sparse index.", chunks_path)
    return Bm25Retriever(chunks, isolation_guardrail)


# RAG injection guardrail used when running the app with "uvicorn app.main:create_app --factory".
# Returns None (a no-op passthrough in the pipeline) rather than crashing startup if the model
# can't be loaded (e.g. no network to download it from the Hugging Face Hub on first run).
def _default_rag_injection_guardrail(threshold: float) -> RagInjectionGuardrail | None:
    try:
        return RagInjectionGuardrail(threshold=threshold)
    except Exception:
        logger.exception("guardrails.rag_injection_detection is on but RagInjectionGuardrail failed to load. Falling back to no check.")
        return None


# Map domain errors to HTTP responses without leaking internals (API keys, system prompts, stack traces, filesystem paths).
def _register_exception_handlers(fastapi_app: FastAPI) -> None:

    def _json_error(status_code: int, detail: str):
        async def handler(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse(status_code=status_code, content={"detail": detail})

        return handler

    fastapi_app.exception_handler(PatientNotFoundError)(_json_error(404, "No matching patient found"))
    fastapi_app.exception_handler(AmbiguousPatientNameError)(_json_error(400, "full_name matches more than one patient"))
    fastapi_app.exception_handler(SessionNotFoundError)(_json_error(404, "Session not found"))
    fastapi_app.exception_handler(InvalidRequestError)(_json_error(400, "Invalid request"))

    @fastapi_app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError) -> JSONResponse:
        logger.exception("Unhandled domain error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @fastapi_app.exception_handler(Exception)
    async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Run the app with: uvicorn app.main:create_app --factory
