"""Sign-in, registration, password recovery and profile pages."""

from __future__ import annotations

import streamlit as st

from services import auth, authz

VIEW_KEY = "auth_view"
LOGIN, REGISTER, FORGOT, RESET = "login", "register", "forgot", "reset"
NOTICE_KEY = "auth_notice"
PROFILE_KEY = "show_profile"


def _go(view: str) -> None:
    st.session_state[VIEW_KEY] = view
    st.rerun()


def _link(label: str, view: str, key: str) -> None:
    if st.button(label, key=key, use_container_width=True):
        _go(view)


def _sign_in_user(user: dict) -> None:
    from ui.landing import sign_in

    token = auth.create_session(user["id"])
    sign_in(user, token)


def _login_form() -> None:
    notice = st.session_state.pop(NOTICE_KEY, None)
    if notice:
        st.success(notice)
    with st.form("auth_login_form", border=False):
        email = st.text_input("Email", key="auth_login_email", autocomplete="username")
        password = st.text_input(
            "Password", type="password", key="auth_login_password",
            autocomplete="current-password",
        )
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        try:
            user = auth.authenticate(email, password)
        except auth.AuthError as exc:
            st.error(str(exc))
        else:
            _sign_in_user(user)
    _link("Forgot password?", FORGOT, "auth_to_forgot")
    _link("Create account", REGISTER, "auth_to_register")
    st.caption("Sessions end when you sign out or after 8 hours.")


def _register_form() -> None:
    st.caption(auth.PASSWORD_GUIDANCE)
    with st.form("auth_register_form", border=False):
        name = st.text_input("Full name", key="auth_reg_name", autocomplete="name")
        email = st.text_input("Email", key="auth_reg_email", autocomplete="email")
        password = st.text_input(
            "Password", type="password", key="auth_reg_password",
            autocomplete="new-password",
        )
        confirm = st.text_input(
            "Confirm password", type="password", key="auth_reg_confirm",
            autocomplete="new-password",
        )
        submitted = st.form_submit_button(
            "Create patient account", type="primary", use_container_width=True
        )
    if submitted:
        try:
            auth.register_patient(name, email, password, confirm)
        except auth.AuthError as exc:
            st.error(str(exc))
        else:
            st.session_state[NOTICE_KEY] = "Account created. Please sign in."
            _go(LOGIN)
    st.caption("Accounts created here are patient accounts. Coordinator accounts are set up by an administrator.")
    _link("Back to sign in", LOGIN, "auth_reg_back")


def _forgot_form() -> None:
    if not auth.email_delivery_configured():
        st.warning(
            "Password recovery is unavailable because email delivery is not "
            "configured. Ask your administrator to reset your password."
        )
    else:
        with st.form("auth_forgot_form", border=False):
            email = st.text_input("Email", key="auth_forgot_email", autocomplete="email")
            submitted = st.form_submit_button(
                "Send reset link", type="primary", use_container_width=True
            )
        if submitted:
            try:
                st.info(auth.request_password_reset(email))
            except auth.AuthError as exc:
                st.error(str(exc))
        _link("I have a reset code", RESET, "auth_to_reset")
    _link("Back to sign in", LOGIN, "auth_forgot_back")


def _reset_form() -> None:
    prefill = st.session_state.pop("auth_reset_prefill", None)
    if prefill:
        st.session_state["auth_reset_token"] = prefill
    st.caption(auth.PASSWORD_GUIDANCE)
    with st.form("auth_reset_form", border=False):
        token = st.text_input("Reset code from your email", type="password", key="auth_reset_token")
        password = st.text_input("New password", type="password", key="auth_reset_password",
                                 autocomplete="new-password")
        confirm = st.text_input("Confirm new password", type="password", key="auth_reset_confirm",
                                autocomplete="new-password")
        submitted = st.form_submit_button("Reset password", type="primary", use_container_width=True)
    if submitted:
        try:
            auth.reset_password(token, password, confirm)
        except auth.AuthError as exc:
            st.error(str(exc))
        else:
            st.session_state.pop("auth_reset_token", None)
            st.session_state[NOTICE_KEY] = "Password updated. Please sign in."
            _go(LOGIN)
    _link("Back to sign in", LOGIN, "auth_reset_back")


_TITLES = {
    LOGIN: ("Sign in", "Use your Clinical Trial AI account."),
    REGISTER: ("Create account", "Register to read approved study information."),
    FORGOT: ("Forgot password", "We will email a single-use reset link."),
    RESET: ("Reset password", "Choose a new password."),
}


def render_auth_panel() -> None:
    """The signed-out panel on the welcome page."""
    token = st.query_params.get("reset_token")
    if token:
        st.session_state["auth_reset_prefill"] = token
        st.session_state[VIEW_KEY] = RESET
        del st.query_params["reset_token"]
    view = st.session_state.get(VIEW_KEY, LOGIN)
    title, subtitle = _TITLES[view]
    st.html(
        f'<div class="ctai-role-heading">{title}</div>'
        f'<p class="ctai-role-subtitle">{subtitle}</p>'
    )
    {LOGIN: _login_form, REGISTER: _register_form,
     FORGOT: _forgot_form, RESET: _reset_form}[view]()


def render_signed_in_panel(user: dict) -> None:
    from ui.landing import ROLE_LABELS, open_workspace, sign_out

    st.html(
        f'<div class="ctai-role-heading">Welcome back, {_escape(user["full_name"])}</div>'
        f'<p class="ctai-role-subtitle">Signed in as {ROLE_LABELS[user["role"]]}.</p>'
    )
    if st.button("Open workspace", key="open_workspace_btn", type="primary",
                 use_container_width=True):
        open_workspace(user)
    if st.button("Log out", key="landing_logout_btn", use_container_width=True):
        sign_out()


def _escape(text: str) -> str:
    import html

    return html.escape(text or "")


def render_profile_page(user: dict) -> None:
    from ui import theme
    from ui.landing import ROLE_LABELS

    theme.page_head("Your account", "Your sign-in details and password.")
    st.markdown(f"**Name:** {user['full_name']}")
    st.markdown(f"**Email:** {user['email']}")
    st.markdown(f"**Role:** {ROLE_LABELS[user['role']]}")
    st.subheader("Change password")
    st.caption(auth.PASSWORD_GUIDANCE)
    with st.form("auth_change_password_form", border=False, clear_on_submit=True):
        current = st.text_input("Current password", type="password", key="auth_cp_current",
                                autocomplete="current-password")
        new = st.text_input("New password", type="password", key="auth_cp_new",
                            autocomplete="new-password")
        confirm = st.text_input("Confirm new password", type="password", key="auth_cp_confirm",
                                autocomplete="new-password")
        submitted = st.form_submit_button("Change password", type="primary")
    if submitted:
        try:
            auth.change_password(user["id"], current, new, confirm)
        except auth.AuthError as exc:
            st.error(str(exc))
        else:
            st.success("Password changed.")
    left, right = st.columns(2)
    if left.button("Back to workspace", key="profile_back_btn"):
        st.session_state[PROFILE_KEY] = False
        st.rerun()
    if right.button("Log out", key="profile_logout_btn"):
        from ui.landing import sign_out

        sign_out()
