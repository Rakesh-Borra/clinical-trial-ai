"""Accounts, sessions, roles, password recovery and the signed-in application."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from services import auth, authz, review_store
from tests.auth_helpers import sign_in_as

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
PASSWORD = "Passw0rd-test-123"


def _user(role="patient", email="pat@example.test"):
    return auth.create_user("Pat Example", email, PASSWORD, role)


def _row(email):
    import sqlite3

    conn = sqlite3.connect(review_store.db_path())
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()


# --- registration, login, hashing ------------------------------------------

def test_register_and_login_roundtrip():
    created = auth.register_patient("Ann Lee", "  Ann@Example.TEST ", PASSWORD, PASSWORD)
    assert created["role"] == "patient"
    assert created["email"] == "ann@example.test"
    assert auth.authenticate("ANN@example.test", PASSWORD)["id"] == created["id"]


def test_public_registration_cannot_choose_coordinator():
    import inspect

    assert "role" not in inspect.signature(auth.register_patient).parameters
    assert auth.register_patient("A B", "a@example.test", PASSWORD, PASSWORD)["role"] == "patient"


def test_duplicate_email_is_rejected_case_insensitively():
    _user(email="dup@example.test")
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.register_patient("Other", "DUP@example.test", PASSWORD, PASSWORD)


@pytest.mark.parametrize("password", ["short1", "onlyletterslong", "1234567890123"])
def test_weak_passwords_are_rejected(password):
    with pytest.raises(auth.AuthError):
        auth.create_user("A", "w@example.test", password)


def test_mismatched_confirmation_is_rejected():
    with pytest.raises(auth.AuthError, match="do not match"):
        auth.register_patient("A", "m@example.test", PASSWORD, PASSWORD + "x")


def test_password_is_stored_as_argon2id_hash():
    _user()
    stored = _row("pat@example.test")["password_hash"]
    assert stored.startswith("$argon2id$")
    assert PASSWORD not in stored


def test_invalid_credentials_share_one_generic_message():
    _user()
    with pytest.raises(auth.AuthError) as wrong:
        auth.authenticate("pat@example.test", "wrong-Password-1")
    with pytest.raises(auth.AuthError) as unknown:
        auth.authenticate("nobody@example.test", "wrong-Password-1")
    assert str(wrong.value) == str(unknown.value) == auth.GENERIC_LOGIN_ERROR


def test_repeated_failures_lock_the_account_even_for_the_right_password():
    _user()
    for _ in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(auth.AuthError):
            auth.authenticate("pat@example.test", "wrong-Password-1")
    with pytest.raises(auth.AuthError, match="Too many"):
        auth.authenticate("pat@example.test", PASSWORD)


# --- sessions ---------------------------------------------------------------

def test_session_token_is_hashed_and_logout_ends_it():
    user = _user()
    token = auth.create_session(user["id"])
    assert auth.user_for_session(token)["email"] == "pat@example.test"
    import sqlite3

    conn = sqlite3.connect(review_store.db_path())
    assert token not in str(conn.execute("SELECT * FROM auth_sessions").fetchall())
    conn.close()
    auth.end_session(token)
    assert auth.user_for_session(token) is None
    assert auth.user_for_session(None) is None
    assert auth.user_for_session("forged") is None


def test_expired_session_is_rejected(monkeypatch):
    user = _user()
    token = auth.create_session(user["id"])
    later = datetime.now(timezone.utc) + timedelta(hours=auth.SESSION_HOURS + 1)
    monkeypatch.setattr(auth, "_now", lambda: later)
    assert auth.user_for_session(token) is None


# --- password change --------------------------------------------------------

def test_change_password_requires_current_password_and_applies_policy():
    user = _user()
    with pytest.raises(auth.AuthError, match="current password"):
        auth.change_password(user["id"], "bad-Password-1", "Another-Pass-99", "Another-Pass-99")
    with pytest.raises(auth.AuthError):
        auth.change_password(user["id"], PASSWORD, "short", "short")
    auth.change_password(user["id"], PASSWORD, "Another-Pass-99", "Another-Pass-99")
    assert auth.authenticate("pat@example.test", "Another-Pass-99")
    with pytest.raises(auth.AuthError):
        auth.authenticate("pat@example.test", PASSWORD)


# --- password recovery ------------------------------------------------------

@pytest.fixture
def mailbox(monkeypatch):
    monkeypatch.setenv("CTAI_SMTP_HOST", "smtp.invalid")
    monkeypatch.setenv("CTAI_SMTP_FROM", "noreply@example.test")
    sent = []
    monkeypatch.setattr(auth, "send_email", lambda to, subject, body: sent.append((to, body)))
    return sent


def _token_from(body):
    return next(l.split(": ", 1)[1] for l in body.splitlines() if l.startswith("Reset code: "))


def test_recovery_is_unavailable_without_email_configuration(monkeypatch):
    monkeypatch.delenv("CTAI_SMTP_HOST", raising=False)
    with pytest.raises(auth.AuthError, match="unavailable"):
        auth.request_password_reset("pat@example.test")


def test_reset_response_does_not_reveal_whether_account_exists(mailbox):
    _user()
    known = auth.request_password_reset("pat@example.test")
    unknown = auth.request_password_reset("ghost@example.test")
    assert known == unknown
    assert [to for to, _ in mailbox] == ["pat@example.test"]


def test_reset_token_is_hashed_single_use_and_ends_sessions(mailbox):
    user = _user()
    session = auth.create_session(user["id"])
    auth.request_password_reset("pat@example.test")
    token = _token_from(mailbox[0][1])
    assert token not in str(dict(_row("pat@example.test")))
    auth.reset_password(token, "Brand-New-Pass-7", "Brand-New-Pass-7")
    assert auth.authenticate("pat@example.test", "Brand-New-Pass-7")
    assert auth.user_for_session(session) is None
    with pytest.raises(auth.AuthError, match="invalid or has expired"):
        auth.reset_password(token, "Third-Password-8", "Third-Password-8")


def test_reset_token_expires(mailbox, monkeypatch):
    _user()
    auth.request_password_reset("pat@example.test")
    token = _token_from(mailbox[0][1])
    later = datetime.now(timezone.utc) + timedelta(minutes=auth.RESET_MINUTES + 1)
    monkeypatch.setattr(auth, "_now", lambda: later)
    with pytest.raises(auth.AuthError, match="invalid or has expired"):
        auth.reset_password(token, "Brand-New-Pass-7", "Brand-New-Pass-7")


def test_new_reset_request_invalidates_the_previous_token(mailbox):
    _user()
    auth.request_password_reset("pat@example.test")
    auth.request_password_reset("pat@example.test")
    with pytest.raises(auth.AuthError):
        auth.reset_password(_token_from(mailbox[0][1]), "Brand-New-Pass-7", "Brand-New-Pass-7")


def test_reset_token_is_never_returned_to_the_caller(mailbox):
    _user()
    notice = auth.request_password_reset("pat@example.test")
    assert _token_from(mailbox[0][1]) not in notice


# --- migration safety -------------------------------------------------------

def test_migration_preserves_existing_review_data_and_is_repeatable():
    review_store.log_action("protocol-x", "created", "keep me")
    _user()
    auth.get_user_by_email("pat@example.test")
    assert review_store.history("protocol-x")[0]["action"] == "created"
    import sqlite3

    conn = sqlite3.connect(review_store.db_path())
    assert conn.execute("SELECT COUNT(*) FROM auth_schema_version").fetchone()[0] == len(auth._MIGRATIONS)
    conn.close()


# --- authorization ----------------------------------------------------------

def test_require_coordinator_rejects_anonymous_and_patients(monkeypatch):
    import streamlit as st

    monkeypatch.setattr(st, "session_state", {})
    with pytest.raises(authz.AuthorizationError):
        authz.require_coordinator()
    sign_in_as(st.session_state, "patient")
    with pytest.raises(authz.AuthorizationError):
        authz.require_coordinator()
    sign_in_as(st.session_state, "coordinator")
    assert authz.require_coordinator()["role"] == "coordinator"


def test_publish_is_refused_for_patients_and_anonymous(monkeypatch):
    import streamlit as st
    import app

    monkeypatch.setattr(st, "session_state", {"coordinator_protocol_id": "p"})
    ok, message = app.publish_current_protocol()
    assert not ok and "sign in" in message.lower()
    sign_in_as(st.session_state, "patient")
    ok, message = app.publish_current_protocol()
    assert not ok and "coordinator" in message
    ok, message = app.unpublish_current_protocol()
    assert not ok and "coordinator" in message


def test_stolen_role_value_does_not_grant_coordinator_access():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at = at.run()
    assert at.session_state["role"] is None
    assert not at.exception


def test_patient_cannot_open_coordinator_workspace_by_forcing_role():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at.session_state["role"] = "coordinator"
    at = at.run()
    assert at.session_state["role"] is None
    assert not [b for b in at.button if b.key == "analyze_btn"]


def test_logged_out_session_loses_workspace():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator")
    at = at.run()
    assert at.session_state["role"] == "coordinator"
    auth.end_session(at.session_state["auth_token"])
    at = at.run()
    assert at.session_state["role"] is None


# --- pages ------------------------------------------------------------------

def _submit(at, label):
    next(b for b in at.button if b.label == label).click()
    return at.run()


def test_register_page_creates_patient_and_redirects_to_login():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button(key="auth_to_register").click()
    at = at.run()
    at.text_input(key="auth_reg_name").set_value("New Person")
    at.text_input(key="auth_reg_email").set_value("new@example.test")
    at.text_input(key="auth_reg_password").set_value(PASSWORD)
    at.text_input(key="auth_reg_confirm").set_value(PASSWORD)
    at = _submit(at, "Create patient account")
    assert auth.get_user_by_email("new@example.test")["role"] == "patient"
    assert "auth_login_email" in {w.key for w in at.text_input}
    assert any("Account created" in s.value for s in at.success)


def test_login_page_shows_friendly_error_for_bad_credentials():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.text_input(key="auth_login_email").set_value("who@example.test")
    at.text_input(key="auth_login_password").set_value("nope")
    at = _submit(at, "Sign in")
    assert any(auth.GENERIC_LOGIN_ERROR in e.value for e in at.error)
    assert at.session_state["role"] is None


def test_forgot_page_says_recovery_unavailable_without_email(monkeypatch):
    monkeypatch.delenv("CTAI_SMTP_HOST", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button(key="auth_to_forgot").click()
    at = at.run()
    assert any("unavailable" in w.value for w in at.warning)
    assert not [b for b in at.button if b.label == "Send reset link"]


def test_profile_page_shows_account_and_changes_password():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    user = sign_in_as(at.session_state, "patient")
    at = at.run()
    at.button(key="profile_btn").click()
    at = at.run()
    text = " ".join(m.value for m in at.markdown)
    assert user["email"] in text and "Patient/caregiver" in text
    at.text_input(key="auth_cp_current").set_value(PASSWORD)
    at.text_input(key="auth_cp_new").set_value("Changed-Pass-42")
    at.text_input(key="auth_cp_confirm").set_value("Changed-Pass-42")
    at = _submit(at, "Change password")
    assert any("Password changed" in s.value for s in at.success)
    assert auth.authenticate(user["email"], "Changed-Pass-42")


def test_admin_cli_creates_coordinator(monkeypatch):
    from scripts import manage_users

    monkeypatch.setattr(manage_users.getpass, "getpass", lambda prompt="": PASSWORD)
    assert manage_users.main(["create-coordinator", "--email", "c@example.test", "--name", "Dr C"]) == 0
    assert auth.get_user_by_email("c@example.test")["role"] == "coordinator"
    assert manage_users.main(["create-coordinator", "--email", "c@example.test", "--name", "Dr C"]) == 1


# --- server-side guards on review operations --------------------------------

_WRITES = [
    lambda: review_store.log_action("p", "x", ""),
    lambda: review_store.unpublish("p"),
    lambda: review_store.approve_summary("p"),
    lambda: review_store.save_summary_draft("p", "text"),
    lambda: review_store.withdraw_summary_approval("p"),
    lambda: review_store.set_coverage_override("p", "risks", "reason"),
    lambda: review_store.clear_coverage_override("p", "risks"),
    lambda: review_store.save_original_summary("p", "text", "ai"),
    lambda: review_store.clear_generated_items("p"),
]


def _script_page(role):
    """Run review_store writes inside a live app session for the given account role."""
    at = AppTest.from_function(_run_writes, default_timeout=30)
    if role:
        sign_in_as(at.session_state, role)
    return at.run()


def _run_writes():
    import streamlit as st
    from services import review_store as rs

    calls = [
        lambda: rs.log_action("p", "x", ""),
        lambda: rs.unpublish("p"),
        lambda: rs.approve_summary("p"),
        lambda: rs.save_summary_draft("p", "text"),
        lambda: rs.withdraw_summary_approval("p"),
        lambda: rs.set_coverage_override("p", "risks", "reason"),
        lambda: rs.clear_coverage_override("p", "risks"),
        lambda: rs.save_original_summary("p", "text", "ai"),
        lambda: rs.clear_generated_items("p"),
    ]
    denied = 0
    for call in calls:
        try:
            call()
        except PermissionError:
            denied += 1
    st.session_state["denied"] = denied


@pytest.mark.parametrize("role", [None, "patient"])
def test_review_writes_are_refused_in_a_live_session_without_coordinator(role):
    assert _script_page(role).session_state["denied"] == len(_WRITES)


def test_review_writes_are_allowed_for_a_signed_in_coordinator():
    assert _script_page("coordinator").session_state["denied"] == 0


def test_every_review_store_writer_is_guarded():
    import inspect

    guarded = {
        name for name, fn in inspect.getmembers(review_store, inspect.isfunction)
        if getattr(fn, "__wrapped__", None) is not None
    }
    assert {"publish", "unpublish", "save_draft", "approve_summary", "log_action",
            "save_summary_draft", "set_coverage_override"} <= guarded


def test_signed_out_welcome_page_has_no_account_menu():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not [b for b in at.button if b.key in ("landing_profile_btn", "landing_logout_menu_btn")]
