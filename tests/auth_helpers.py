"""Create real accounts and sessions for tests of signed-in behaviour."""

from __future__ import annotations

from services import auth, authz


def sign_in_as(session_state, role: str, email: str | None = None) -> dict:
    """Create a user with `role`, open a real session and store its token."""
    address = email or f"{role}@example.test"
    user = auth.get_user_by_email(address, role) or auth.create_user(
        f"Test {role.title()}", address, "Passw0rd-test-123", role
    )
    session_state[authz.TOKEN_KEY] = auth.create_session(user["id"])
    session_state["role"] = role
    return user
