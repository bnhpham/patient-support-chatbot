"""
RAG injection guardrail - llm-semantic-router/toolcall-sentinel, a pretrained ModernBERT-base
injection classifier, run on a patient's own retrieved chunks.

Runs on the retrieved chunks, after retrieval and before the prompt is built. Any chunk scoring
above threshold on the model's INJECTION_RISK label is dropped from context (and logged) rather
than blocking the whole turn.

DEMO SCOPE, READ BEFORE TRUSTING THIS IN ANYTHING BEYOND A DEMO: the default threshold (0.8) comes
from a narrow, single-patient calibration (James Whitfield's 28 chunks) where it cleanly separated
two of the three poison payloads with a real margin. Treat 0.8 as a demo illustration, not a validated
production threshold. Also note: `overdose` (a poisoned dosage recommendation with no imperative
injection phrasing) has not separated from real content on ANY classifier tested this project.

Toggled by guardrails.rag_injection_detection in config/rag.yaml. When off, this guardrail is
never constructed and the pipeline node is a no-op passthrough.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.rag.types import ScoredChunk

logger = logging.getLogger("app.guardrails.rag_injection")

DEFAULT_MODEL_REPO = "llm-semantic-router/toolcall-sentinel"
INJECTION_LABEL_INDEX = 1  # config.json: id2label = {"0": "SAFE", "1": "INJECTION_RISK"}
MAX_LENGTH = 512


class RagInjectionGuardrail:
    def __init__(self, model_repo: str = DEFAULT_MODEL_REPO, threshold: float = 0.8) -> None:

        self._tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_repo)
        self._model.eval()
        self._threshold = threshold

    # Drop chunks whose INJECTION_RISK probability exceeds threshold. One batched forward pass for all chunks, not one per chunk.
    def enforce(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:

        if not chunks:
            return chunks

        encoded = self._tokenizer([sc.chunk.text for sc in chunks], truncation=True, max_length=MAX_LENGTH,
                                  padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**encoded).logits
        scores = torch.softmax(logits, dim=-1)[:, INJECTION_LABEL_INDEX].tolist()

        kept, dropped = [], []
        for sc, score in zip(chunks, scores):
            if score > self._threshold:
                dropped.append((sc, score))
            else:
                kept.append(sc)

        if dropped:
            logger.warning("RagInjectionGuardrail dropped %d chunk(s) as injection-risk outliers: %s",
                           len(dropped),
                           [(sc.chunk.chunk_id, round(score, 3)) for sc, score in dropped])

        return kept
