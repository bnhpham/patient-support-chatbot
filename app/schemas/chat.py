"""Request/response schemas for the /sessions/{session_id}/chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)


class SourceRef(BaseModel):
    source_record_id: str
    section: str


class ChatResponse(BaseModel):
    message_id: str
    response: str
    sources: list[SourceRef]
