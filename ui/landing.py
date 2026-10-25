"""
Welcome page: pick a role and sign in.

Sign-in uses real accounts (services/auth.py). The role of the signed-in
account, never a user-chosen option, decides which workspace opens.
"""

from __future__ import annotations

import streamlit as st

from services import auth, authz
from ui import auth_pages, theme

ROLE_KEY = "role"
NAME_KEY = "user_name"
HOME_BUTTON_KEY = "home_btn"

COORDINATOR = "coordinator"
PATIENT = "patient"
ROLE_LABELS = {COORDINATOR: "Research coordinator", PATIENT: "Patient/caregiver"}
PAGE_HISTORY_KEY = "page_history"
PAGE_HISTORY_INDEX_KEY = "page_history_index"


def home_page() -> dict[str, object]:
    return {"role": None}


def coordinator_page(step: str, max_step: int) -> dict[str, object]:
    return {"role": COORDINATOR, "step": step, "max_step": max_step}


def patient_page(
    protocol_id: str | None = None,
    *,
    summary: bool = False,
) -> dict[str, object]:
    return {
        "role": PATIENT,
        "protocol_id": protocol_id,
        "summary": summary,
    }


def _ensure_page_history() -> None:
    if PAGE_HISTORY_KEY not in st.session_state:
        st.session_state[PAGE_HISTORY_KEY] = [home_page()]
        st.session_state[PAGE_HISTORY_INDEX_KEY] = 0


def _restore_page(page: dict[str, object]) -> None:
    st.session_state[ROLE_KEY] = page.get("role")
    if page.get("role") == COORDINATOR:
        st.session_state["coord_step"] = page["step"]
        st.session_state["coord_max_step"] = max(
            st.session_state.get("coord_max_step", 0),
            int(page["max_step"]),
        )
    elif page.get("role") == PATIENT:
        protocol_id = page.get("protocol_id")
        if protocol_id != st.session_state.get("patient_active_protocol_id"):
            st.session_state["patient_history"] = []
            st.session_state.pop("patient_chat_input", None)
            st.session_state["patient_active_protocol_id"] = protocol_id
        st.session_state["patient_protocol_select"] = protocol_id
        st.session_state["patient_summary_open"] = bool(page.get("summary"))
    else:
        st.session_state[NAME_KEY] = ""


def record_page_visit(page: dict[str, object]) -> None:
    """Record a rendered page transition once, dropping any forward history."""
    _ensure_page_history()
    history = st.session_state[PAGE_HISTORY_KEY]
    index = st.session_state[PAGE_HISTORY_INDEX_KEY]
    current = history[index]
    same_page = current.get("role") == page.get("role")
    if page.get("role") == COORDINATOR:
        same_page = same_page and current.get("step") == page.get("step")
    elif page.get("role") == PATIENT:
        same_page = same_page and current.get("summary") == page.get("summary")
        if page.get("summary"):
            same_page = same_page and current.get("protocol_id") == page.get("protocol_id")
    if same_page:
        history[index] = page.copy()
        return
    del history[index + 1 :]
    history.append(page.copy())
    st.session_state[PAGE_HISTORY_INDEX_KEY] = len(history) - 1


def navigate_to_page(page: dict[str, object]) -> None:
    """Visit a page and restore its route-specific state."""
    record_page_visit(page)
    _restore_page(page)


def navigate_history(offset: int) -> bool:
    """Move within browser-style page history and restore the destination."""
    _ensure_page_history()
    history = st.session_state[PAGE_HISTORY_KEY]
    index = st.session_state[PAGE_HISTORY_INDEX_KEY] + offset
    if index < 0 or index >= len(history):
        return False
    st.session_state[PAGE_HISTORY_INDEX_KEY] = index
    _restore_page(history[index])
    return True


def _can_navigate_history(offset: int) -> bool:
    _ensure_page_history()
    index = st.session_state[PAGE_HISTORY_INDEX_KEY]
    target = index + offset
    return 0 <= target < len(st.session_state[PAGE_HISTORY_KEY])


def _render_history_arrows(previous, next_) -> None:
    if previous.button(
        "←",
        key="nav_previous",
        disabled=not _can_navigate_history(-1),
    ):
        if navigate_history(-1):
            st.rerun()
    if next_.button(
        "→",
        key="nav_next",
        disabled=not _can_navigate_history(1),
    ):
        if navigate_history(1):
            st.rerun()


def current_role() -> str | None:
    return st.session_state.get(ROLE_KEY)


def open_workspace(user: dict) -> None:
    page = (
        coordinator_page("1 · Upload", 0)
        if user["role"] == COORDINATOR
        else patient_page()
    )
    navigate_to_page(page)
    st.session_state[NAME_KEY] = user["full_name"]
    st.rerun()


def sign_in(user: dict, token: str) -> None:
    """Start an authenticated session for a user who just proved their credentials."""
    st.session_state[authz.TOKEN_KEY] = token
    open_workspace(user)


def open_profile(user: dict) -> None:
    """Open the profile page in the workspace that matches the account."""
    navigate_to_page(coordinator_page("1 · Upload", 0) if user["role"] == COORDINATOR else patient_page())
    st.session_state["show_profile"] = True
    st.session_state[NAME_KEY] = user["full_name"]
    st.rerun()


def go_home() -> None:
    navigate_to_page(home_page())
    st.session_state["show_profile"] = False
    st.rerun()


def sign_out() -> None:
    """End the server-side session and clear everything tied to the user."""
    auth.end_session(st.session_state.pop(authz.TOKEN_KEY, None))
    st.session_state["show_profile"] = False
    st.session_state[NAME_KEY] = ""
    navigate_to_page(home_page())
    st.rerun()


def render_top_bar(
    role: str | None = None,
) -> None:
    """Show the Home action and the signed-in user's account menu."""
    _ensure_page_history()
    with st.container(key="topbar"):
        left, previous, next_, _, right = st.columns(
            [2, .5, .5, 6, 1.3],
            vertical_alignment="center",
        )
        if left.button(
            "Clinical Trial AI",
            key=HOME_BUTTON_KEY,
            icon=":material/health_and_safety:",
        ):
            go_home()

        _render_history_arrows(previous, next_)

        user = authz.current_user()
        if role and user:
            with right.popover("Account", use_container_width=True):
                st.caption(f"{user['full_name']} · {ROLE_LABELS[user['role']]}")
                if st.button("Profile", key="profile_btn", use_container_width=True):
                    st.session_state["show_profile"] = True
                    st.rerun()
                if st.button("Logout", key="logout_btn", use_container_width=True):
                    sign_out()


_RESEARCH_VISUAL = """
<svg class="ctai-research-art" viewBox="0 0 520 190" role="img"
     aria-label="Abstract clinical research illustration">
  <defs>
    <linearGradient id="researchLine" x1="0" x2="1">
      <stop offset="0" stop-color="#8ca9b8" stop-opacity=".08"/>
      <stop offset=".5" stop-color="#c4d9e2" stop-opacity=".54"/>
      <stop offset="1" stop-color="#8ca9b8" stop-opacity=".08"/>
    </linearGradient>
    <radialGradient id="researchGlow">
      <stop stop-color="#7899a7" stop-opacity=".16"/>
      <stop offset="1" stop-color="#7899a7" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <ellipse cx="268" cy="98" rx="238" ry="86" fill="url(#researchGlow)"/>
  <path d="M22 100h112l26-35 33 67 36-90 31 58h238" fill="none"
        stroke="url(#researchLine)" stroke-width="1.4"/>
  <path d="M37 143h94l26-20 24 20h100l24-21 26 21h149" fill="none"
        stroke="#a9c9d6" stroke-opacity=".15" stroke-width="1"/>
  <circle cx="160" cy="65" r="4" fill="#b7d0da" fill-opacity=".75"/>
  <circle cx="229" cy="42" r="3" fill="#b7d0da" fill-opacity=".55"/>
  <circle cx="286" cy="100" r="4" fill="#b7d0da" fill-opacity=".7"/>
  <circle cx="340" cy="122" r="3" fill="#b7d0da" fill-opacity=".45"/>
  <rect x="364" y="27" width="78" height="78" rx="20" fill="none"
        stroke="#c2d8e0" stroke-opacity=".17" transform="rotate(12 403 66)"/>
  <rect x="382" y="46" width="42" height="42" rx="12" fill="none"
        stroke="#c2d8e0" stroke-opacity=".3" transform="rotate(12 403 67)"/>
  <path d="M403 55v24M391 67h24" stroke="#c2d8e0" stroke-opacity=".56"
        stroke-width="1.4" stroke-linecap="round"/>
</svg>
"""


def render_landing() -> None:
    theme.inject_landing_css()
    record_page_visit(home_page())

    with st.container(key="landing_shell"):
        with st.container(key="landing_nav"):
            brand, previous, next_, _, account = st.columns(
                [2, .5, .5, 6, 1.3],
                vertical_alignment="center",
            )
            if brand.button(
                "Clinical Trial AI",
                key=HOME_BUTTON_KEY,
                icon=":material/health_and_safety:",
            ):
                navigate_to_page(home_page())
                st.rerun()
            _render_history_arrows(previous, next_)
            user = authz.current_user()
            if user:
                with account.popover("Account", key="landing_account", use_container_width=True):
                    st.caption(f"{user['full_name']} · {ROLE_LABELS[user['role']]}")
                    if st.button("Profile", key="landing_profile_btn", use_container_width=True):
                        open_profile(user)
                    if st.button("Logout", key="landing_logout_menu_btn", use_container_width=True):
                        sign_out()

        with st.container(key="landing_hero"):
            left, right = st.columns([1.12, .88], gap="large", vertical_alignment="center")
            with left:
                st.html(
                    '<section id="landing-home" class="ctai-landing-copy">'
                    '<div class="ctai-landing-kicker">Clinical research, made clearer</div>'
                    '<h1 class="ctai-landing-title">Welcome to'
                    '<span>Clinical Trial AI</span></h1>'
                    '<p class="ctai-landing-description">'
                    "Understand clinical trial protocols with AI-assisted "
                    "summaries, source-page references, and human-reviewed "
                    "patient information."
                    "</p>"
                    '<div class="ctai-trust-list">'
                    '<div class="ctai-trust-item"><span class="ctai-trust-dot"></span>'
                    "Protocol details stay connected to their source pages</div>"
                    '<div class="ctai-trust-item"><span class="ctai-trust-dot"></span>'
                    "Clinical content is reviewed before it is shared</div>"
                    '<div class="ctai-trust-item"><span class="ctai-trust-dot"></span>'
                    "Patient guidance includes only coordinator-approved information</div>"
                    "</div>"
                    + _RESEARCH_VISUAL
                    + "</section>"
                )

            with right:
                with st.container(key="landing_role_panel"):
                    user = authz.current_user()
                    if user:
                        auth_pages.render_signed_in_panel(user)
                    else:
                        auth_pages.render_auth_panel()

        st.html(
            '<div class="ctai-landing-footer">'
            "Research prototype for educational demonstration only · "
            "Not medical advice or an eligibility decision"
            "</div>"
        )
