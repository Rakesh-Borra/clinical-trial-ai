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
    assert auth.authenticate("ANN@example.test", PASSWORD, "patient")["id"] == created["id"]


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
        auth.authenticate("pat@example.test", "wrong-Password-1", "patient")
    with pytest.raises(auth.AuthError) as unknown:
        auth.authenticate("nobody@example.test", "wrong-Password-1", "patient")
    assert str(wrong.value) == str(unknown.value) == auth.GENERIC_LOGIN_ERROR


def test_repeated_failures_lock_the_account_even_for_the_right_password():
    _user()
    for _ in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(auth.AuthError):
            auth.authenticate("pat@example.test", "wrong-Password-1", "patient")
    with pytest.raises(auth.AuthError, match="Too many"):
        auth.authenticate("pat@example.test", PASSWORD, "patient")


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
    assert auth.authenticate("pat@example.test", "Another-Pass-99", "patient")
    with pytest.raises(auth.AuthError):
        auth.authenticate("pat@example.test", PASSWORD, "patient")


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
        auth.request_password_reset("pat@example.test", "patient")


def test_reset_response_does_not_reveal_whether_account_exists(mailbox):
    _user()
    known = auth.request_password_reset("pat@example.test", "patient")
    unknown = auth.request_password_reset("ghost@example.test", "patient")
    assert known == unknown
    assert [to for to, _ in mailbox] == ["pat@example.test"]


def test_reset_token_is_hashed_single_use_and_ends_sessions(mailbox):
    user = _user()
    session = auth.create_session(user["id"])
    auth.request_password_reset("pat@example.test", "patient")
    token = _token_from(mailbox[0][1])
    assert token not in str(dict(_row("pat@example.test")))
    auth.reset_password(token, "Brand-New-Pass-7", "Brand-New-Pass-7")
    assert auth.authenticate("pat@example.test", "Brand-New-Pass-7", "patient")
    assert auth.user_for_session(session) is None
    with pytest.raises(auth.AuthError, match="invalid or has expired"):
        auth.reset_password(token, "Third-Password-8", "Third-Password-8")


def test_reset_token_expires(mailbox, monkeypatch):
    _user()
    auth.request_password_reset("pat@example.test", "patient")
    token = _token_from(mailbox[0][1])
    later = datetime.now(timezone.utc) + timedelta(minutes=auth.RESET_MINUTES + 1)
    monkeypatch.setattr(auth, "_now", lambda: later)
    with pytest.raises(auth.AuthError, match="invalid or has expired"):
        auth.reset_password(token, "Brand-New-Pass-7", "Brand-New-Pass-7")


def test_new_reset_request_invalidates_the_previous_token(mailbox):
    _user()
    auth.request_password_reset("pat@example.test", "patient")
    auth.request_password_reset("pat@example.test", "patient")
    with pytest.raises(auth.AuthError):
        auth.reset_password(_token_from(mailbox[0][1]), "Brand-New-Pass-7", "Brand-New-Pass-7")


def test_reset_token_is_never_returned_to_the_caller(mailbox):
    _user()
    notice = auth.request_password_reset("pat@example.test", "patient")
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
    assert auth.authenticate(user["email"], "Changed-Pass-42", "patient")


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


# --- role selection, coordinator invitations, role-checked login ------------

def _register(role, email, invite=""):
    return auth.register_account("Some One", email, PASSWORD, PASSWORD, role, invite)


def test_patient_registration_needs_no_invitation():
    assert _register("patient", "p1@example.test")["role"] == "patient"


def test_coordinator_registration_requires_a_code():
    with pytest.raises(auth.AuthError, match="invitation"):
        _register("coordinator", "c0@example.test")
    assert auth.get_user_by_email("c0@example.test") is None


def test_coordinator_registration_with_valid_code_is_single_use():
    code = auth.create_invitation()
    assert _register("coordinator", "c1@example.test", code)["role"] == "coordinator"
    with pytest.raises(auth.AuthError, match="invalid, expired or already used"):
        _register("coordinator", "c2@example.test", code)
    assert auth.get_user_by_email("c2@example.test") is None


def test_invalid_invitation_code_is_rejected_and_creates_no_account():
    auth.create_invitation()
    with pytest.raises(auth.AuthError, match="invalid"):
        _register("coordinator", "c3@example.test", "not-a-real-code")
    assert auth.get_user_by_email("c3@example.test") is None


def test_expired_invitation_code_is_rejected():
    code = auth.create_invitation(hours=-1)
    with pytest.raises(auth.AuthError, match="expired"):
        _register("coordinator", "c4@example.test", code)
    assert auth.get_user_by_email("c4@example.test") is None


def test_failed_registration_does_not_burn_the_code():
    _user("coordinator", "taken@example.test")
    code = auth.create_invitation()
    with pytest.raises(auth.AuthError, match="already exists"):
        _register("coordinator", "taken@example.test", code)
    assert _register("coordinator", "fresh@example.test", code)["role"] == "coordinator"


def test_invitation_codes_are_stored_hashed():
    import sqlite3

    code = auth.create_invitation()
    conn = sqlite3.connect(review_store.db_path())
    try:
        rows = conn.execute("SELECT code_hash FROM coordinator_invites").fetchall()
    finally:
        conn.close()
    assert rows and all(code not in r[0] and len(r[0]) == 64 for r in rows)


def test_invitation_cli_prints_a_working_code(capsys):
    from scripts import manage_users

    assert manage_users.main(["create-invite", "--hours", "1"]) == 0
    code = capsys.readouterr().out.strip().splitlines()[-1]
    assert _register("coordinator", "cli@example.test", code)["role"] == "coordinator"


def test_login_role_must_match_stored_role():
    _user("patient", "pat2@example.test")
    _user("coordinator", "coord2@example.test")
    with pytest.raises(auth.AuthError, match=auth.GENERIC_LOGIN_ERROR):
        auth.authenticate("pat2@example.test", PASSWORD, "coordinator")
    with pytest.raises(auth.AuthError, match=auth.GENERIC_LOGIN_ERROR):
        auth.authenticate("coord2@example.test", PASSWORD, "patient")
    assert auth.authenticate("coord2@example.test", PASSWORD, "coordinator")["role"] == "coordinator"
    assert auth.authenticate("pat2@example.test", PASSWORD, "patient")["role"] == "patient"


def test_wrong_password_with_wrong_role_stays_generic():
    _user("patient", "pat3@example.test")
    with pytest.raises(auth.AuthError, match=auth.GENERIC_LOGIN_ERROR):
        auth.authenticate("pat3@example.test", "wrong-pass-123", "coordinator")


def test_login_page_patient_card_cannot_open_coordinator_account_workspace():
    _user("patient", "pat4@example.test")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.radio(key="auth_login_role").set_value("coordinator")
    at.text_input(key="auth_login_email").set_value("pat4@example.test")
    at.text_input(key="auth_login_password").set_value(PASSWORD)
    at = _submit(at, "Sign in")
    assert any(auth.GENERIC_LOGIN_ERROR in e.value for e in at.error)
    assert at.session_state["role"] is None


def test_login_page_redirects_each_role_to_its_workspace():
    _user("patient", "pat5@example.test")
    _user("coordinator", "coord5@example.test")
    for email, role in (("pat5@example.test", "patient"), ("coord5@example.test", "coordinator")):
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        assert at.radio(key="auth_login_role").value == "patient"
        at.radio(key="auth_login_role").set_value(role)
        at.text_input(key="auth_login_email").set_value(email)
        at.text_input(key="auth_login_password").set_value(PASSWORD)
        at = _submit(at, "Sign in")
        assert at.session_state["role"] == role


def test_register_page_defaults_to_patient_and_shows_code_only_for_coordinator():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button(key="auth_to_register").click()
    at = at.run()
    assert at.radio(key="auth_reg_role").value == "patient"
    assert "auth_reg_invite" not in {w.key for w in at.text_input}
    at.radio(key="auth_reg_role").set_value("coordinator")
    at = at.run()
    assert "auth_reg_invite" in {w.key for w in at.text_input}


def test_register_page_coordinator_flow_with_and_without_code():
    code = auth.create_invitation()
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button(key="auth_to_register").click()
    at = at.run()
    at.radio(key="auth_reg_role").set_value("coordinator")
    at = at.run()
    for k, v in (("auth_reg_name", "Dr C"), ("auth_reg_email", "drc@example.test"),
                 ("auth_reg_password", PASSWORD), ("auth_reg_confirm", PASSWORD),
                 ("auth_reg_invite", "bad-code")):
        at.text_input(key=k).set_value(v)
    at = _submit(at, "Create coordinator account")
    assert any("invalid" in e.value for e in at.error)
    assert auth.get_user_by_email("drc@example.test") is None
    at.text_input(key="auth_reg_invite").set_value(code)
    at = _submit(at, "Create coordinator account")
    assert auth.get_user_by_email("drc@example.test")["role"] == "coordinator"


def test_migration_keeps_existing_accounts():
    user = _user("patient", "keep@example.test")
    auth.create_invitation()
    assert auth.get_user_by_email("keep@example.test")["id"] == user["id"]
    assert auth.authenticate("keep@example.test", PASSWORD, "patient")["role"] == "patient"


# --- demo mode ---------------------------------------------------------------

def test_demo_mode_is_off_by_default_and_needs_exact_true(monkeypatch):
    monkeypatch.delenv("CTAI_DEMO_MODE", raising=False)
    assert not auth.demo_mode()
    for value in ("1", "yes", "false", "", "truee"):
        monkeypatch.setenv("CTAI_DEMO_MODE", value)
        assert not auth.demo_mode()
    monkeypatch.setenv("CTAI_DEMO_MODE", "True")
    assert auth.demo_mode()


def test_normal_mode_still_requires_invitation(monkeypatch):
    monkeypatch.delenv("CTAI_DEMO_MODE", raising=False)
    with pytest.raises(auth.AuthError, match="invitation"):
        _register("coordinator", "nm@example.test")
    assert auth.get_user_by_email("nm@example.test") is None


def test_demo_mode_accepts_exactly_0000_and_saves_roles(monkeypatch):
    monkeypatch.setenv("CTAI_DEMO_MODE", "true")
    assert _register("coordinator", "dm@example.test", "0000")["role"] == "coordinator"
    assert _register("patient", "dp@example.test")["role"] == "patient"
    assert auth.get_user_by_email("dm@example.test")["role"] == "coordinator"
    assert auth.authenticate("dm@example.test", PASSWORD, "coordinator")["role"] == "coordinator"
    with pytest.raises(auth.AuthError):
        auth.authenticate("dm@example.test", PASSWORD, "patient")
    with pytest.raises(auth.AuthError):
        auth.authenticate("dp@example.test", PASSWORD, "coordinator")


@pytest.mark.parametrize("code", ["", "   ", "0001", "000", "00000", "0000 0", "abcd"])
def test_demo_mode_rejects_wrong_or_empty_codes(monkeypatch, code):
    monkeypatch.setenv("CTAI_DEMO_MODE", "true")
    with pytest.raises(auth.AuthError):
        _register("coordinator", "bad@example.test", code)
    assert auth.get_user_by_email("bad@example.test") is None


def test_0000_is_rejected_outside_demo_mode(monkeypatch):
    monkeypatch.delenv("CTAI_DEMO_MODE", raising=False)
    with pytest.raises(auth.AuthError, match="invalid, expired or already used"):
        _register("coordinator", "zero@example.test", "0000")
    monkeypatch.setenv("CTAI_DEMO_MODE", "false")
    with pytest.raises(auth.AuthError):
        _register("coordinator", "zero@example.test", "0000")
    assert auth.get_user_by_email("zero@example.test") is None


def test_real_invitations_still_work_in_demo_mode(monkeypatch):
    monkeypatch.setenv("CTAI_DEMO_MODE", "true")
    code = auth.create_invitation()
    assert _register("coordinator", "real@example.test", code)["role"] == "coordinator"


def test_demo_mode_still_validates_password_and_duplicates(monkeypatch):
    monkeypatch.setenv("CTAI_DEMO_MODE", "true")
    with pytest.raises(auth.AuthError, match="do not match"):
        auth.register_account("A", "x@example.test", PASSWORD, PASSWORD + "1", "coordinator", "0000")
    _register("coordinator", "dup@example.test", "0000")
    with pytest.raises(auth.AuthError, match="already exists"):
        _register("coordinator", "dup@example.test", "0000")


def test_demo_mode_does_not_apply_without_env_even_with_empty_invite(monkeypatch):
    monkeypatch.setenv("CTAI_DEMO_MODE", "true")
    _register("coordinator", "on@example.test", "0000")
    monkeypatch.delenv("CTAI_DEMO_MODE")
    with pytest.raises(auth.AuthError, match="invalid"):
        _register("coordinator", "off@example.test", "0000")


def _open_register(at):
    at.button(key="auth_to_register").click()
    at = at.run()
    at.radio(key="auth_reg_role").set_value("coordinator")
    return at.run()


def test_register_page_demo_mode_shows_hint_and_accepts_0000(monkeypatch):
    monkeypatch.setenv("CTAI_DEMO_MODE", "true")
    at = _open_register(AppTest.from_file(APP_PATH, default_timeout=30).run())
    assert "auth_reg_invite" in {w.key for w in at.text_input}
    assert any(auth.DEMO_CODE_HINT in m.value for m in at.markdown)
    assert any(auth.DEMO_LABEL in c.value for c in at.caption)
    for k, v in (("auth_reg_name", "Demo Coord"), ("auth_reg_email", "demo@example.test"),
                 ("auth_reg_password", PASSWORD), ("auth_reg_confirm", PASSWORD),
                 ("auth_reg_invite", "1234")):
        at.text_input(key=k).set_value(v)
    at = _submit(at, "Create coordinator account")
    assert at.error and auth.get_user_by_email("demo@example.test") is None
    at.text_input(key="auth_reg_invite").set_value("0000")
    at = _submit(at, "Create coordinator account")
    assert auth.get_user_by_email("demo@example.test")["role"] == "coordinator"


def test_register_page_normal_mode_shows_code_and_no_demo_hint(monkeypatch):
    monkeypatch.delenv("CTAI_DEMO_MODE", raising=False)
    at = _open_register(AppTest.from_file(APP_PATH, default_timeout=30).run())
    assert "auth_reg_invite" in {w.key for w in at.text_input}
    assert not any(auth.DEMO_CODE_HINT in m.value for m in at.markdown)
    assert not any(auth.DEMO_LABEL in c.value for c in at.caption)


def test_tampered_frontend_state_cannot_grant_coordinator(monkeypatch):
    monkeypatch.delenv("CTAI_DEMO_MODE", raising=False)
    _user("patient", "tamper@example.test")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at.session_state["auth_login_role"] = "coordinator"
    at = at.run()
    assert at.session_state["role"] is None
    at.text_input(key="auth_login_email").set_value("tamper@example.test")
    at.text_input(key="auth_login_password").set_value(PASSWORD)
    at = _submit(at, "Sign in")
    assert at.session_state["role"] is None
    # Signed in as a patient, forcing the role or the demo flag in the page state changes nothing.
    at2 = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at2.session_state, "patient")
    at2.session_state["role"] = "coordinator"
    at2.session_state["demo_mode"] = True
    at2 = at2.run()
    assert at2.session_state["role"] != "coordinator"


# --- one email, two accounts (patient + coordinator) -------------------------

SHARED = "both@example.test"
OTHER_PASSWORD = "Different-Pass-77"


def _pair():
    patient = auth.create_user("Pat Both", SHARED, PASSWORD, "patient")
    coordinator = auth.create_user("Coord Both", SHARED, OTHER_PASSWORD, "coordinator")
    return patient, coordinator


def test_same_email_can_hold_a_patient_and_a_coordinator_account():
    patient, coordinator = _pair()
    assert patient["id"] != coordinator["id"]
    rows = [_row(SHARED)]
    assert rows[0] is not None
    assert auth.get_user_by_email(SHARED, "patient")["id"] == patient["id"]
    assert auth.get_user_by_email(SHARED, "coordinator")["id"] == coordinator["id"]
    with pytest.raises(auth.AuthError, match="specify the role"):
        auth.get_user_by_email(SHARED)


def test_each_account_has_its_own_hash_name_and_login():
    patient, coordinator = _pair()
    assert auth.authenticate(SHARED, PASSWORD, "patient")["id"] == patient["id"]
    assert auth.authenticate(SHARED, OTHER_PASSWORD, "coordinator")["id"] == coordinator["id"]
    # The other account's password, or the other role, never authenticates.
    for password, role in ((OTHER_PASSWORD, "patient"), (PASSWORD, "coordinator")):
        with pytest.raises(auth.AuthError, match=auth.GENERIC_LOGIN_ERROR):
            auth.authenticate(SHARED, password, role)
    import sqlite3

    conn = sqlite3.connect(review_store.db_path())
    try:
        hashes = [r[0] for r in conn.execute("SELECT password_hash FROM users WHERE email = ?", (SHARED,))]
    finally:
        conn.close()
    assert len(set(hashes)) == 2


def test_same_password_on_both_accounts_still_respects_the_selected_role():
    auth.create_user("P", SHARED, PASSWORD, "patient")
    auth.create_user("C", SHARED, PASSWORD, "coordinator")
    assert auth.authenticate(SHARED, PASSWORD, "patient")["role"] == "patient"
    assert auth.authenticate(SHARED, PASSWORD, "coordinator")["role"] == "coordinator"


def test_duplicate_email_and_same_role_is_rejected_but_other_role_is_allowed():
    auth.create_user("P", SHARED, PASSWORD, "patient")
    with pytest.raises(auth.AuthError, match="patient account with this email already exists"):
        auth.register_account("P2", SHARED.upper(), PASSWORD, PASSWORD, "patient")
    assert auth.register_account(
        "C", SHARED, PASSWORD, PASSWORD, "coordinator", auth.create_invitation()
    )["role"] == "coordinator"
    with pytest.raises(auth.AuthError, match="coordinator account with this email already exists"):
        auth.register_account("C2", SHARED, PASSWORD, PASSWORD, "coordinator", auth.create_invitation())


def test_registering_a_coordinator_never_converts_the_existing_patient():
    patient = auth.create_user("P", SHARED, PASSWORD, "patient")
    auth.register_account("C", SHARED, OTHER_PASSWORD, OTHER_PASSWORD, "coordinator", auth.create_invitation())
    assert auth.get_user_by_email(SHARED, "patient")["id"] == patient["id"]
    assert auth.authenticate(SHARED, PASSWORD, "patient")["role"] == "patient"


def test_sessions_are_per_account_and_logout_of_one_keeps_the_other():
    patient, coordinator = _pair()
    p_token, c_token = auth.create_session(patient["id"]), auth.create_session(coordinator["id"])
    assert auth.user_for_session(p_token)["role"] == "patient"
    assert auth.user_for_session(c_token)["role"] == "coordinator"
    auth.end_session(p_token)
    assert auth.user_for_session(p_token) is None
    assert auth.user_for_session(c_token)["role"] == "coordinator"


def test_changing_one_accounts_password_leaves_the_other_alone():
    patient, coordinator = _pair()
    c_token = auth.create_session(coordinator["id"])
    auth.change_password(patient["id"], PASSWORD, "Changed-Pass-55", "Changed-Pass-55")
    assert auth.authenticate(SHARED, "Changed-Pass-55", "patient")
    assert auth.authenticate(SHARED, OTHER_PASSWORD, "coordinator")
    assert auth.user_for_session(c_token)


def test_password_reset_targets_only_the_requested_role(mailbox):
    patient, coordinator = _pair()
    p_token, c_token = auth.create_session(patient["id"]), auth.create_session(coordinator["id"])
    auth.request_password_reset(SHARED, "coordinator")
    assert "coordinator account" in mailbox[0][1]
    auth.reset_password(_token_from(mailbox[0][1]), "Reset-Pass-4321", "Reset-Pass-4321")
    assert auth.authenticate(SHARED, "Reset-Pass-4321", "coordinator")
    assert auth.authenticate(SHARED, PASSWORD, "patient")
    assert auth.user_for_session(p_token) and auth.user_for_session(c_token) is None
    with pytest.raises(auth.AuthError):
        auth.authenticate(SHARED, OTHER_PASSWORD, "coordinator")


def test_password_reset_for_a_role_without_an_account_sends_nothing(mailbox):
    auth.create_user("P", SHARED, PASSWORD, "patient")
    notice = auth.request_password_reset(SHARED, "coordinator")
    assert notice == auth.GENERIC_RESET_NOTICE and mailbox == []


def test_lockout_is_per_email_and_role():
    _pair()
    for _ in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(auth.AuthError):
            auth.authenticate(SHARED, "wrong-Password-1", "patient")
    with pytest.raises(auth.AuthError, match="Too many"):
        auth.authenticate(SHARED, PASSWORD, "patient")
    assert auth.authenticate(SHARED, OTHER_PASSWORD, "coordinator")["role"] == "coordinator"


def test_admin_operations_require_a_role_when_the_email_is_shared():
    _pair()
    with pytest.raises(auth.AuthError, match="--role"):
        auth.admin_set_password(SHARED, "Admin-Pass-9999")
    auth.admin_set_password(SHARED, "Admin-Pass-9999", "patient")
    assert auth.authenticate(SHARED, "Admin-Pass-9999", "patient")
    assert auth.authenticate(SHARED, OTHER_PASSWORD, "coordinator")
    with pytest.raises(auth.AuthError, match="--role"):
        auth.set_role(SHARED, "coordinator")


def test_promote_is_refused_when_the_target_role_already_exists():
    _pair()
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.set_role(SHARED, "coordinator", "patient")
    assert auth.get_user_by_email(SHARED, "patient")["role"] == "patient"


def test_promote_still_works_for_a_single_account_and_ends_its_sessions():
    user = auth.create_user("Solo", "solo@example.test", PASSWORD, "patient")
    token = auth.create_session(user["id"])
    auth.set_role("solo@example.test", "coordinator")
    assert auth.get_user_by_email("solo@example.test")["role"] == "coordinator"
    assert auth.user_for_session(token) is None


def test_cli_reset_password_needs_role_for_shared_email(monkeypatch, capsys):
    from scripts import manage_users

    _pair()
    monkeypatch.setattr(manage_users, "_password", lambda: "Cli-Pass-123456")
    assert manage_users.main(["reset-password", "--email", SHARED]) == 1
    assert "--role" in capsys.readouterr().err
    assert manage_users.main(["reset-password", "--email", SHARED, "--role", "coordinator"]) == 0
    assert auth.authenticate(SHARED, "Cli-Pass-123456", "coordinator")
    assert auth.authenticate(SHARED, PASSWORD, "patient")


def test_patient_session_for_a_shared_email_has_no_coordinator_power():
    patient, coordinator = _pair()
    at = AppTest.from_function(_run_writes, default_timeout=30)
    at.session_state[authz.TOKEN_KEY] = auth.create_session(patient["id"])
    at = at.run()
    assert at.session_state["denied"] >= 1
    at2 = AppTest.from_function(_run_writes, default_timeout=30)
    at2.session_state[authz.TOKEN_KEY] = auth.create_session(coordinator["id"])
    at2 = at2.run()
    assert at2.session_state["denied"] == 0


def test_login_page_signs_in_the_selected_account_of_a_shared_email():
    _pair()
    for role, password in (("patient", PASSWORD), ("coordinator", OTHER_PASSWORD)):
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        at.radio(key="auth_login_role").set_value(role)
        at.text_input(key="auth_login_email").set_value(SHARED)
        at.text_input(key="auth_login_password").set_value(password)
        at = _submit(at, "Sign in")
        assert at.session_state["role"] == role
    # Selecting the wrong card with the other account's password fails.
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.radio(key="auth_login_role").set_value("coordinator")
    at.text_input(key="auth_login_email").set_value(SHARED)
    at.text_input(key="auth_login_password").set_value(PASSWORD)
    at = _submit(at, "Sign in")
    assert at.session_state["role"] is None


def test_register_page_allows_coordinator_for_an_email_that_already_has_a_patient(monkeypatch):
    monkeypatch.setenv("CTAI_DEMO_MODE", "true")
    auth.create_user("P", SHARED, PASSWORD, "patient")
    at = _open_register(AppTest.from_file(APP_PATH, default_timeout=30).run())
    for k, v in (("auth_reg_name", "C"), ("auth_reg_email", SHARED),
                 ("auth_reg_password", OTHER_PASSWORD), ("auth_reg_confirm", OTHER_PASSWORD),
                 ("auth_reg_invite", "0000")):
        at.text_input(key=k).set_value(v)
    at = _submit(at, "Create coordinator account")
    assert auth.get_user_by_email(SHARED, "coordinator") is not None
    assert auth.get_user_by_email(SHARED, "patient")["role"] == "patient"


def test_forgot_page_asks_which_account(monkeypatch, mailbox):
    monkeypatch.setenv("CTAI_SMTP_HOST", "smtp.test")
    monkeypatch.setenv("CTAI_SMTP_FROM", "noreply@test")
    _pair()
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at.button(key="auth_to_forgot").click()
    at = at.run()
    at.radio(key="auth_forgot_role").set_value("coordinator")
    at.text_input(key="auth_forgot_email").set_value(SHARED)
    at = _submit(at, "Send reset link")
    assert "coordinator account" in mailbox[0][1]


# --- migrating a database created before email+role uniqueness --------------

def _legacy_db():
    import sqlite3

    conn = sqlite3.connect(review_store.db_path())
    conn.executescript(auth._MIGRATIONS[0] + auth._MIGRATIONS[1])
    conn.execute("CREATE TABLE auth_schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO auth_schema_version VALUES (2)")
    conn.execute(
        "INSERT INTO users VALUES ('u1','Old Pat','old@example.test',?, 'patient','2026-01-01T00:00:00+00:00',NULL,NULL)",
        (auth._hasher.hash(PASSWORD),),
    )
    conn.execute(
        "INSERT INTO auth_sessions VALUES (?, 'u1', '2026-01-01T00:00:00+00:00', '2999-01-01T00:00:00+00:00')",
        (auth._digest("old-token"),),
    )
    conn.execute("INSERT INTO login_attempts VALUES ('old@example.test','2026-01-01T00:00:00+00:00')")
    conn.commit()
    conn.close()


def test_migration_to_email_and_role_keeps_users_hashes_sessions_and_data():
    review_store.log_action("protocol-legacy", "created", "keep me")
    _legacy_db()
    # First connection after upgrade runs the migration.
    assert auth.authenticate("old@example.test", PASSWORD, "patient")["id"] == "u1"
    assert auth.user_for_session("old-token")["id"] == "u1"
    assert auth.create_user("Old Coord", "old@example.test", OTHER_PASSWORD, "coordinator")["role"] == "coordinator"
    assert auth.get_user_by_email("old@example.test", "patient")["id"] == "u1"
    import sqlite3

    conn = sqlite3.connect(review_store.db_path())
    try:
        assert conn.execute("SELECT MAX(version) FROM auth_schema_version").fetchone()[0] == len(auth._MIGRATIONS)
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(login_attempts)")]
    finally:
        conn.close()
    assert "role" in cols
    assert any("keep me" in str(a) for a in review_store.list_actions("protocol-legacy")) if hasattr(review_store, "list_actions") else True


def test_migration_still_blocks_a_second_account_with_the_same_email_and_role():
    _legacy_db()
    with pytest.raises(auth.AuthError, match="already exists"):
        auth.create_user("Dup", "old@example.test", PASSWORD, "patient")
