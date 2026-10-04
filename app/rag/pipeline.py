"""
Runtime RAG pipeline: a small LangGraph StateGraph.

retrieval -> (RAG injection check) -> prompt_build -> generate -> validate_output
                                                          ^              |
                                                          +--------------+
                                                          (regenerate on a flagged answer, up to
                                                          generation.max_regenerations, then fall
                                                          back to a fixed safe message)

Kept intentionally small - this is also the natural seam for inserting future guardrail nodes without restructuring the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config import RagConfig
from app.llm.claude_client import LLMClient
from app.prompts.utils import load_prompt, render_prompt
from app.rag import prompt_builder
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.types import ScoredChunk

logger = logging.getLogger("app.rag.pipeline")

CORRECTION_TEMPLATE = load_prompt("regeneration_correction.txt")
FALLBACK_ANSWER = load_prompt("fallback_answer.txt")


class PipelineState(TypedDict, total=False):
    patient_id: str
    question: str
    history: list[dict]
    candidates: list[ScoredChunk]
    system: str
    messages: list[dict]
    answer: str
    correction: str | None
    regen_attempts: int


@dataclass
class PipelineResult:
    answer: str
    chunks_used: list[ScoredChunk]


class RagPipeline:
    def __init__(self, hybrid_retriever: HybridRetriever, llm_client: LLMClient, rag_config: RagConfig,
                 medication_guardrail=None, rag_injection_guardrail=None,
                 hallucination_guardrail=None, policy_guardrail=None) -> None:

        self._hybrid_retriever = hybrid_retriever
        self._llm_client = llm_client
        self._config = rag_config
        self._medication_guardrail = medication_guardrail
        self._rag_injection_guardrail = rag_injection_guardrail
        self._hallucination_guardrail = hallucination_guardrail
        self._policy_guardrail = policy_guardrail
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("rag_injection_check", self._rag_injection_check_node)
        graph.add_node("context_build", self._prompt_build_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("validate_output", self._validate_output_node)

        graph.set_entry_point("retrieval")
        graph.add_edge("retrieval", "rag_injection_check")
        graph.add_edge("rag_injection_check", "context_build")
        graph.add_edge("context_build", "generate")
        graph.add_edge("generate", "validate_output")
        graph.add_conditional_edges("validate_output", self._route_after_validate, {"generate": "generate", END: END})

        return graph.compile()

    def run(self, question: str, patient_id: str, history: list[dict]) -> PipelineResult:
        initial_state: PipelineState = {"patient_id": patient_id,
                                        "question": question,
                                        "history": history}

        final_state = self._graph.invoke(initial_state)

        return PipelineResult(
            answer=final_state["answer"],
            chunks_used=final_state.get("candidates", []),
        )


    #============================================================================
    # Langchain Stategraph Nodes
    #============================================================================

    # Retrieval
    def _retrieval_node(self, state: PipelineState) -> dict:

        retrieval_cfg = self._config.retrieval
        candidates = self._hybrid_retriever.retrieve(sub_queries=[state["question"]],
                                                     patient_id=state["patient_id"],
                                                     dense_weight=retrieval_cfg.dense_weight,
                                                     sparse_weight=retrieval_cfg.sparse_weight,
                                                     candidate_top_k=retrieval_cfg.candidate_top_k)

        return {"candidates": candidates}

    # RAG injection guardrail: drop retrieved chunks flagged as injection risks (see app/guardrails/rag_injection.py).
    def _rag_injection_check_node(self, state: PipelineState) -> dict:

        # Skip the check if guardrail is set off
        if not self._config.guardrails.rag_injection_detection or self._rag_injection_guardrail is None:
            return {"candidates": state["candidates"]}

        # Detect injected chunks
        checked = self._rag_injection_guardrail.enforce(state["candidates"])
        if len(checked) != len(state["candidates"]):
            logger.info("rag_injection_check: %d candidates -> %d", len(state["candidates"]), len(checked))

        return {"candidates": checked}

    # Build prompt
    def _prompt_build_node(self, state: PipelineState) -> dict:

        # Medication guardrail: When on, add the trusted drug reference and any detected dosage conflict.
        fact_check = None
        if self._config.guardrails.fact_check and self._medication_guardrail is not None:
            fact_check = self._medication_guardrail.reference_block(state["candidates"])
            if fact_check:
                logger.info("fact_check: reference block added to system prompt")

        prompt = prompt_builder.build_prompt(question=state["question"],
                                              chunks=state["candidates"],
                                              history=state.get("history", []),
                                              fact_check=fact_check)

        return {"system": prompt.system, "messages": prompt.messages}

    # Send prompt to LLM API and return response.
    def _generate_node(self, state: PipelineState) -> dict:

        temperature = self._config.generation.temperature
        messages = state["messages"]
        correction = state.get("correction")

        # Append flagged violations as a trailing user turn, so the model can see what to fix without re-running retrieval
        if correction:
            correction_message = render_prompt(CORRECTION_TEMPLATE, violation=correction)
            messages += [{"role": "user", "content": correction_message}]

        answer = self._llm_client.chat(messages, system=state["system"], temperature=temperature)
        return {"answer": answer}

    # Hallucination & policy guardrails:
    # Both are LLM judges over the drafted answer. A flagged answer is regenerated with the violation fed back (see _generate_node).
    def _validate_output_node(self, state: PipelineState) -> dict:

        violations: list[str] = []

        # Hallucination guardrail
        if self._config.guardrails.hallucination_check and self._hallucination_guardrail is not None:
            reason = self._hallucination_guardrail.check(state["answer"], state.get("candidates", []))
            if reason:
                violations.append(reason)

        # Policy guardrail
        if self._config.guardrails.policy_check and self._policy_guardrail is not None:
            reason = self._policy_guardrail.check(state["answer"])
            if reason:
                violations.append(reason)

        if not violations:
            return {"correction": None}

        # Let the chatbot self-correct its answer up to "max_attempts"-many times before resorting to a fallback answer
        attempts = state.get("regen_attempts", 0)
        max_attempts = self._config.generation.max_regenerations
        if attempts >= max_attempts:
            logger.warning("validate_output: max regenerations (%d) reached, returning fallback answer. violations=%s", max_attempts, violations)
            return {"answer": FALLBACK_ANSWER, "correction": None}

        logger.info("validate_output: answer flagged (attempt %d/%d): %s", attempts + 1, max_attempts, violations)
        return {"correction": "; ".join(violations), "regen_attempts": attempts + 1}

    def _route_after_validate(self, state: PipelineState) -> str:
        return "generate" if state.get("correction") else END
