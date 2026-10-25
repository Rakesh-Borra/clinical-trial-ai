"""Review page section: add a statement the AI did not extract."""

from __future__ import annotations

from typing import Any, Mapping

import streamlit as st

from models.plain_language_models import PatientGuide
from services import authz
from services.manual_statements import (
    CATEGORIES,
    ELIGIBILITY_CHOICES,
    SECTION_ATTR,
    ManualStatementError,
    add_manual_statement,
)

_OPEN = "add_stmt_open"
_RESET = "add_stmt_reset"
_MESSAGE = "add_stmt_message"
_FIELDS = ("add_stmt_category", "add_stmt_eligibility", "add_stmt_text", "add_stmt_page")


def _close_and_clear() -> None:
    for key in _FIELDS:
        st.session_state.pop(key, None)
    st.session_state[_OPEN] = False


def _remove(guide: PatientGuide, item: Any) -> None:
    getattr(guide, SECTION_ATTR[item.section]).items.remove(item)


def render_add_statement(
    guide: PatientGuide, pipeline_result: Mapping[str, Any], persist
) -> None:
    """Button plus compact form; `persist(action, detail)` saves the draft."""
    if st.session_state.pop(_RESET, False):
        _close_and_clear()
    message = st.session_state.pop(_MESSAGE, None)
    if message:
        st.success(message)

    if not st.session_state.get(_OPEN):
        if st.button("+ Add New Statement", key="add_stmt_open_btn"):
            st.session_state[_OPEN] = True
            st.rerun()
        return

    pages = pipeline_result.get("source_pages") or {}
    with st.container(border=True):
        st.markdown("**Add a statement**")
        category = st.selectbox("Category", CATEGORIES, key="add_stmt_category")
        if category == "Eligibility":
            st.radio(
                "Eligibility type",
                list(ELIGIBILITY_CHOICES),
                horizontal=True,
                key="add_stmt_eligibility",
            )
        text = st.text_area("Statement text", height=100, key="add_stmt_text")
        page = st.number_input(
            "Source page in the protocol PDF",
            min_value=1,
            step=1,
            value=1,
            key="add_stmt_page",
        )
        if not pages:
            st.caption(
                "The protocol text is not available here, so this statement "
                "cannot be checked against its source and will be marked unverified."
            )
        save_col, cancel_col, _ = st.columns([1, 1, 3])
        if cancel_col.button("Cancel", key="add_stmt_cancel_btn"):
            _close_and_clear()
            st.rerun()
        if save_col.button("Save Statement", type="primary", key="add_stmt_save_btn"):
            authz.require_coordinator()
            try:
                item = add_manual_statement(
                    guide,
                    category,
                    text,
                    int(page),
                    pages,
                    st.session_state.get("add_stmt_eligibility", "Who can join"),
                )
            except ManualStatementError as exc:
                st.error(str(exc))
                return
            if not persist("statement_added", f"{category}: {item.plain_text}"):
                _remove(guide, item)
                return
            verified = item.source_status == "verified"
            st.session_state[_MESSAGE] = (
                "Statement added as Pending. "
                + (
                    "It matches the cited protocol page; review and approve it."
                    if verified
                    else "It is not verified against the protocol: "
                    + (item.source_issue or "")
                )
            )
            st.session_state[_RESET] = True
            st.rerun()
