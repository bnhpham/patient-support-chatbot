"""
POST /sessions, GET/DELETE /sessions/{id}, POST /sessions/{id}/chat.

Routers stay thin: validate input via Pydantic, call the service layer,
return the response schema. Domain errors raised by the service layer
propagate to the central exception handlers registered in app.main.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_service, get_session_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.session import ConversationMessage, SessionCreateRequest, SessionCreateResponse, SessionInfoResponse
from app.services.chat_service import ChatService
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


# The moment the user accesses the homepage of web interface, create new session.
# New session is created even when refreshing the page
@router.post("", response_model=SessionCreateResponse)
def create_session(body: SessionCreateRequest, session_service: SessionService = Depends(get_session_service)
                   ) -> SessionCreateResponse:
    
    session = session_service.create_session(body.full_name)
    return SessionCreateResponse(session_id=session.session_id, full_name=session.full_name)


# Return chat history
@router.get("/{session_id}", response_model=SessionInfoResponse)
def get_session(session_id: str, session_service: SessionService = Depends(get_session_service)
                ) -> SessionInfoResponse:
    
    session = session_service.get_session(session_id)
    return SessionInfoResponse(session_id=session.session_id,
                               full_name=session.full_name,
                               messages=[ConversationMessage(**m) for m in session.messages])


# Delete chat
@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str, session_service: SessionService = Depends(get_session_service)) -> None:
    session_service.delete_session(session_id)


# Pass the user's chat message to the LLM and return new response back to frontend
@router.post("/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, body: ChatRequest, chat_service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    return chat_service.handle_chat(session_id, body.message)
