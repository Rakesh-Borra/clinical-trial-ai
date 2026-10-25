"""Server-side authorization checks based on the signed-in session, not on visible controls."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from services import auth

TOKEN_KEY = "auth_token"


class AuthorizationError(PermissionError):
    """Raised when the signed-in user may not perform an operation."""


def current_user() -> Optional[Dict[str, Any]]:
    """The user for this browser session; re-checked against the database every call."""
    return auth.user_for_session(st.session_state.get(TOKEN_KEY))


def require_role(role: str) -> Dict[str, Any]:
    user = current_user()
    if user is None:
        raise AuthorizationError("Please sign in to continue.")
    if user["role"] != role:
        raise AuthorizationError(f"This action requires a {role} account.")
    return user


def require_coordinator() -> Dict[str, Any]:
    return require_role(auth.COORDINATOR)
