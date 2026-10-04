"""
Request/response schemas for /sessions endpoints.

Request models use extra="forbid" so that clients cannot smuggle extra
fields (e.g. a client-supplied "patient_id") into the request body - such
requests fail validation with 422 instead of being silently ignored or,
worse, silently honored.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(min_length=1)


class SessionCreateResponse(BaseModel):
    session_id: str
    full_name: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class SessionInfoResponse(BaseModel):
    session_id: str
    full_name: str
    messages: list[ConversationMessage]
