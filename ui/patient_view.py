"""
Patient workspace.

Left: the study guide in plain language (approved statements only).
Right: a fixed, resizable study assistant (see ui/split_pane.py). It is a
normal keyed Streamlit container, so chat history and callbacks are unchanged.

Everything shown here comes from the Patient Guide items a coordinator
approved. Nothing is read from the raw protocol analysis.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from agents.guide_patient_agent import answer_patient_from_guide
from agents.plain_language_agent import display_term, explain_term_query
from models.plain_language_models import PatientGuide, page_label
from services.chat_intents import social_answer
from services.summary_review import format_summary_markdown
from services.guide_review import patient_visible_guide, with_coordinator_referral
from ui import chat_store, split_pane, theme
from ui.landing import (
    NAME_KEY,
    PATIENT,
    patient_page,
    record_page_visit,
    render_top_bar,
)
from ui.patient_guide_tab import is_fully_approved_snapshot, term_definition_answer

_SECTION_MATERIAL_ICON = {
    "Who may be able to join": ":material/groups:",
    "Who may not be able to join": ":material/block:",
    "Trial visits": ":material/calendar_month:",
    "Tests and procedures": ":material/biotech:",
    "Safety information": ":material/health_and_safety:",
}

SUGGESTIONS = [
    "Who can join this study?",
    "Who cannot join?",
    "What visits are there?",
    "What tests will I have?",
    "What does HbA1c mean?",
]

_SECTION_ICON = {
    "Who may be able to join": theme.TEAL,
    "Who may not be able to join": theme.CORAL,
}


def ask(query: str, guide: PatientGuide) -> None:
    """Answer one patient question and add it to the chat history."""
    query = (query or "").strip()
    if not query:
        return
    guide = patient_visible_guide(guide)
    # Whole-message greetings only; anything else keeps the full safety path.
    answer = social_answer(query)
    if answer is not None:
        _record_turn(query, answer, guide)
        return
    approved_terms = {
        term.term.casefold()
        for item in guide.all_items()
        if item.approved and not item.rejected
        for term in item.terms_explained
    }
    queried_terms = explain_term_query(query)
    answer = (
        term_definition_answer(query)
        if queried_terms
        and all(term.term.casefold() in approved_terms for term in queried_terms)
        else None
    )
    if answer is None:
        answer = with_coordinator_referral(
            answer_patient_from_guide(
                query,
                guide,
                history=st.session_state.patient_history,
            )
        )
    _record_turn(query, answer, guide)


def _record_turn(query: str, answer, guide: PatientGuide) -> None:
    st.session_state.patient_history.append({"query": query, "answer": answer})
    chat_store.sync_active(st.session_state.get("patient_active_protocol_id"), guide)


def version_chip(record: dict) -> str:
    """'Version 2.0 · Published 2026-10-24', only from what is actually stored."""
    label = record.get("version_label")
    if not label:
        return ""
    text = f"Version {label}"
    published = str(record.get("published_at") or "")[:10]
    if len(published) == 10:
        text += f" · Published {published}"
    return theme.chip(text, "grey")


def _published_protocols() -> dict[str, dict]:
    """Return only complete, fully approved snapshots saved by publishing."""
    published = st.session_state.get("published_protocols", {})
    visible = {}
    for protocol_id, record in published.items():
        guide = record.get("guide")
        name = record.get("name")
        if (
            isinstance(name, str)
            and name.strip()
            and is_fully_approved_snapshot(guide)
        ):
            visible[protocol_id] = record
    return visible


# ---------------------------------------------------------------------------
# Left: study guide
# ---------------------------------------------------------------------------

_EMPTY_SVG = (
    '<svg width="84" height="84" viewBox="0 0 84 84">'
    '<circle cx="42" cy="42" r="40" fill="#E9F4F1"/>'
    '<rect x="24" y="18" width="36" height="46" rx="8" fill="#fff" stroke="#183A43" stroke-width="4"/>'
    '<rect x="31" y="30" width="22" height="4" rx="2" fill="#9FCBC3"/><rect x="31" y="40" width="16" height="4" rx="2" fill="#9FCBC3"/>'
    '<circle cx="58" cy="58" r="12" fill="#F2C14E"/><path d="M58 51 v8 l5 3" fill="none" stroke="#183A43" stroke-width="3" stroke-linecap="round"/></svg>'
)


def _guide_empty_state() -> None:
    st.html(
        '<div class="ctai-empty">'
        + theme.svg_img(_EMPTY_SVG, "Guide not published yet", "width:84px;height:84px") +
        '<h3>Your study guide is on its way</h3>'
        '<p>The study team has not published the guide yet. You can still ask the assistant on the right, '
        'but it can only share information after the study team approves it.</p></div>'
    )


def _used_terms(guide: PatientGuide) -> List[tuple[str, str]]:
    approved_terms = {
        term.term
        for item in guide.all_items()
        if item.approved
        for term in item.terms_explained
    }
    return [(t.term, t.explanation) for t in guide.glossary if t.term in approved_terms]


def render_guide_panel(guide: Optional[PatientGuide]) -> None:
    if guide is not None and guide.analysis_complete is not True:
        st.warning(
            "This protocol analysis is incomplete or could not be verified. "
            "The patient guide is unavailable until analysis is complete."
        )
        return
    if guide is None or guide.approved_count() == 0:
        _guide_empty_state()
        return

    complete = guide.is_fully_approved()
    status = theme.chip("Approved by your study team", "teal") if complete else theme.chip("Some parts still being reviewed", "sun")
    proto = f'<span class="ctai-chip grey">Protocol {theme.e(guide.protocol_id)}</span>' if guide.protocol_id else ""
    st.html(
        '<div class="ctai-guide-head"><div>'
        f'<div class="ctai-label">Your study guide</div><h2>{theme.e(guide.trial_title or "Clinical trial")}</h2>'
        '<p class="ctai-guide-sub">A plain language guide to help you understand this clinical trial.</p></div>'
        f'<div class="ctai-chip-row">{proto}{status}</div></div>'
    )

    if guide.summary:
        with st.expander("Summary", expanded=True, icon=":material/summarize:"):
            st.markdown(format_summary_markdown(guide.summary))

    for section in guide.sections():
        color = _SECTION_ICON.get(section.title, theme.LAVENDER)
        with st.expander(
            section.title,
            expanded=True,
            icon=_SECTION_MATERIAL_ICON.get(section.title, ":material/article:"),
        ):
            st.html(f'<p style="margin:0 0 .3rem">{theme.e(section.introduction)}</p>')
            if section.unavailable_message:
                st.caption(section.unavailable_message)
                continue
            approved = [i for i in section.items if i.approved]
            pending = len(section.items) - len(approved)
            if not approved:
                st.html('<div class="ctai-pending">This part is still being reviewed by the study team.</div>')
                continue
            for item in approved:
                page = f"<small>Protocol {page_label(item)}</small>" if item.source_page else ""
                st.html(f'<div class="ctai-item">{theme.e(item.plain_text)}{page}</div>')
            if pending:
                word = "item is" if pending == 1 else "items are"
                st.html(f'<div class="ctai-pending">{pending} more {word} still being reviewed by the study team and not shown yet.</div>')

    terms = _used_terms(guide)
    if terms:
        chips = "".join(
            f'<div class="ctai-term"><b>{theme.e(display_term(term))}</b> · {theme.e(expl)}</div>' for term, expl in terms
        )
        st.html(f'<div class="ctai-guide-sec"><h3>Words to know</h3></div><div class="ctai-terms">{chips}</div>')

    st.caption(guide.disclaimer)


# ---------------------------------------------------------------------------
# Right: chat
# ---------------------------------------------------------------------------

def _render_chat_toolbar(guide: PatientGuide) -> None:
    """New chat and history icons, laid out beside the window controls by CSS."""
    protocol_id = st.session_state.get("patient_active_protocol_id")
    with st.container(key="chat_toolbar", horizontal=True, gap="xxsmall"):
        if st.button(
            "New chat",
            key="new_chat_btn",
            icon=":material/edit_square:",
            help="New chat",
            type="tertiary",
        ):
            chat_store.new_chat(protocol_id, guide)
            st.rerun()
        with st.popover(
            "Chat history",
            icon=":material/history:",
            help="Chat history",
            type="tertiary",
            key="chat_history_popover",
        ):
            chats = chat_store.list_chats(protocol_id, guide)
            if not chats:
                st.caption("No saved conversations yet.")
            active_id = st.session_state.get(chat_store.ACTIVE_KEY)
            for chat in chats:
                count = len(chat["messages"])
                label = f'{chat["title"]} · {count} message{"s" if count != 1 else ""}'
                if st.button(
                    label,
                    key=f'history_{chat["id"]}',
                    disabled=chat["id"] == active_id
                    and bool(st.session_state.patient_history),
                    use_container_width=True,
                ):
                    if chat_store.open_chat(protocol_id, guide, chat["id"]):
                        st.rerun()


def render_chat_panel(guide: PatientGuide) -> None:
    st.html(split_pane.CONTROLS_HTML)
    _render_chat_toolbar(guide)
    st.html('<div class="ctai-label">Study assistant</div><h3 class="ctai-chat-title">Ask a question</h3>')

    with st.container(key="chat_messages", border=False):
        st.html(
            '<div class="ctai-chat-hello"><b>Hi! 👋</b> I can explain your study guide in everyday words. '
            'I only share what your study team has approved. I can\'t give medical advice or decide whether you can join.</div>'
        )
        for i, item in enumerate(st.session_state.patient_history):
            with st.container(key=f"msg_user_{i}", border=False):
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(item["query"])
            with st.container(key=f"msg_assistant_{i}", border=False):
                with st.chat_message("assistant", avatar=":material/medical_services:"):
                    st.markdown(item["answer"].answer_text)

    # Suggestions only help before the first patient message of a conversation.
    if not st.session_state.patient_history:
        st.caption("Try one of these:")
        cols = st.columns(2)
        for i, text in enumerate(SUGGESTIONS):
            if cols[i % 2].button(text, key=f"suggest_{i}"):
                ask(text, guide)
                st.rerun()

    typed = st.chat_input("Type your question…", key="patient_chat_input")
    if typed:
        ask(typed, guide)
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_patient_workspace() -> None:
    protocols = _published_protocols()
    protocol_ids = list(protocols)
    selected_id = st.session_state.get("patient_protocol_select")
    if selected_id not in protocols:
        selected_id = None
    if not protocol_ids:
        selected_id = None
        st.session_state.pop("patient_protocol_select", None)

    if selected_id != st.session_state.get("patient_active_protocol_id"):
        st.session_state.patient_history = []
        st.session_state.patient_active_protocol_id = selected_id
        st.session_state.patient_summary_open = False
        st.session_state.pop("patient_chat_input", None)

    summary_open = bool(st.session_state.get("patient_summary_open", False))
    record_page_visit(patient_page(selected_id, summary=summary_open))
    render_top_bar(PATIENT)

    name = st.session_state.get(NAME_KEY) or ""
    if not protocols:
        theme.page_head("Select Your Clinical Trial Protocol")
        st.info("No published clinical trial protocols are available yet.")
        return

    theme.page_head(
        "Select Your Clinical Trial Protocol",
        "Choose a protocol published by the study team.",
    )
    selected_id = st.selectbox(
        "Clinical trial protocol",
        options=protocol_ids,
        key="patient_protocol_select",
        format_func=lambda protocol_id: protocols[protocol_id]["name"],
        index=None,
        placeholder="Choose a published protocol",
    )
    if selected_id != st.session_state.get("patient_active_protocol_id"):
        st.session_state.patient_history = []
        st.session_state.patient_active_protocol_id = selected_id
        st.session_state.patient_summary_open = False
        st.session_state.pop("patient_chat_input", None)

    if selected_id is None:
        st.button(
            "View Protocol Summary",
            type="primary",
            key="view_protocol_summary_btn",
            disabled=True,
        )
        return

    if not st.session_state.get("patient_summary_open"):
        if st.button(
            "View Protocol Summary",
            type="primary",
            key="view_protocol_summary_btn",
        ):
            record_page_visit(patient_page(selected_id, summary=True))
            st.session_state.patient_summary_open = True
            st.rerun()
        return

    protocol = protocols[selected_id]
    guide = patient_visible_guide(protocol["guide"])
    st.html(
        '<div class="ctai-chip-row" style="justify-content:flex-start;margin-bottom:.6rem">'
        f'{theme.chip(protocol["name"], "lav")}{version_chip(protocol)}</div>'
    )
    theme.page_head(
        f"Welcome, {name}" if name else "Welcome to your study guide",
        "Read your guide here. Ask the study assistant when you have a question.",
    )

    with st.container(key="guide_panel"):
        render_guide_panel(guide)
    # Out of the normal page flow: positioned by ui/split_pane.py CSS and script.
    with st.container(key="chat_panel"):
        render_chat_panel(guide)
    split_pane.render_script()
