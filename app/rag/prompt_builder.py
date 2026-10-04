"""
Assembles the final "stuff" prompt: system prompt + patient context +
conversation history + current question. Ready for LLMClient.chat().

The system text is returned separately from the message list because the
Claude API takes it as a top-level `system` parameter rather than as a
message with role="system". Keeping it as its own field also gives later
system-prompt-protection guardrails one obvious place to hook, instead of a
string buried inside a message list.

Deliberately no adversarial defense language here - this baseline's system
prompt is intentionally simple; see app/guardrails for the pluggable seams
where later prompt-hardening or context-sanitization would be inserted.

The wording and layout of the system message live in
app/prompts/system_prompt.txt, so the baseline prompt can be swapped for a
hardened one without editing this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.prompts.utils import load_prompt, render_prompt
from app.rag.types import ScoredChunk

logger = logging.getLogger("app.rag.prompt_builder")

# Loaded once at import: a missing template fails at startup, not mid-chat.
SYSTEM_PROMPT_TEMPLATE = load_prompt("system_prompt.txt")
USER_MESSAGE_TEMPLATE = load_prompt("user_message_template.txt")


@dataclass(frozen=True)
class Prompt:
    system: str
    messages: list[dict]


def build_prompt(question: str, chunks: list[ScoredChunk], history: list[dict],
                 fact_check: str | None = None) -> Prompt:

    # System = trusted instructions only. The retrieved records are untrusted
    # (they can be poisoned), so they go into the user message below rather than
    # into the high-authority system prompt.
    system = SYSTEM_PROMPT_TEMPLATE

    # Chunks from the RAG pipeline
    context_text = "\n\n".join(f"[{sc.chunk.section} | {sc.chunk.source_record_id}] {sc.chunk.text}" for sc in chunks
                               ) or "(no matching patient context found)"

    # Current user turn: delimited context + the optional fact-check block +
    # the actual question, all in one user message, built from
    # app/prompts/user_message_template.txt so the retrieved (untrusted)
    # context stays visually delimited from the question via <context> tags.
    fact_check_slot = f"\n<fact_check>\n{fact_check}\n</fact_check>\n" if fact_check else ""
    
    current_user_message = render_prompt(USER_MESSAGE_TEMPLATE, context=context_text,
                                         fact_check=fact_check_slot, question=question)

    # History holds only user/assistant turns and always opens with a user turn,
    # so appending the current turn keeps the alternation the Messages API expects.
    messages: list[dict] = [{"role": turn["role"], "content": turn["content"]} for turn in history]
    messages.append({"role": "user", "content": current_user_message})

    logger.debug("prompt_builder: system=%s messages=%s", system, messages)
    return Prompt(system=system, messages=messages)
