"""FastAPI dependency accessors.

Services are attached to app.state by app.main.create_app(); routers pull
them out via these thin functions instead of constructing anything
themselves.
"""

from __future__ import annotations

from fastapi import Request

from app.services.chat_service import ChatService
from app.services.session_service import SessionService


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service
