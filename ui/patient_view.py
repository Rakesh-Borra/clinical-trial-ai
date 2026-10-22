"""
Patient workspace.

Left: the study guide in plain language (approved statements only).
Right: a chat assistant. The chat box can be made narrower or wider.

Everything shown here comes from the Patient Guide items a coordinator
approved. Nothing is read from the raw protocol analysis.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from agents.guide_patient_agent import answer_patient_from_guide
from agents.plain_language_agent import display_term
from models.plain_language_models import PatientGuide
from ui import theme
from ui.landing import NAME_KEY, PATIENT, render_top_bar
from ui.patient_guide_tab import get_guide, term_definition_answer

CHAT_SIZES = ["Narrow", "Balanced", "Wide"]
_CHAT_RATIOS = {"Narrow": [2.4, 1], "Balanced": [1.5, 1], "Wide": [1, 1]}

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


def ask(query: str) -> None:
    """Answer one patient question and add it to the chat history."""
    query = (query or "").strip()
    if not query:
        return
    answer = term_definition_answer(query)
    if answer is None:
        # Only coordinator-approved guide items are ever used for answers.
        answer = answer_patient_from_guide(query, get_guide())
    st.session_state.patient_history.append({"query": query, "answer": answer})


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
    if guide is None or guide.approved_count() == 0:
        _guide_empty_state()
        return

    complete = guide.is_fully_approved()
    status = theme.chip("Approved by your study team", "teal") if complete else theme.chip("Some parts still being reviewed", "sun")
    proto = f'<span class="ctai-chip grey">Protocol {theme.e(guide.protocol_id)}</span>' if guide.protocol_id else ""
    st.html(
        '<div class="ctai-guide-head"><div>'
        f'<div class="ctai-label">Your study guide</div><h2>{theme.e(guide.trial_title or "Clinical trial")}</h2></div>'
        f'<div class="ctai-chip-row">{proto}{status}</div></div>'
    )

    for section in guide.sections():
        color = _SECTION_ICON.get(section.title, theme.LAVENDER)
        st.html(
            f'<div class="ctai-guide-sec"><span class="ctai-dot" style="background:{color};width:14px;height:14px"></span>'
            f'<h3>{theme.e(section.title)}</h3></div>'
            f'<p style="margin:0 0 .3rem">{theme.e(section.introduction)}</p>'
        )
        if section.unavailable_message:
            st.caption(section.unavailable_message)
            continue
        approved = [i for i in section.items if i.approved]
        pending = len(section.items) - len(approved)
        if not approved:
            st.html('<div class="ctai-pending">This part is still being reviewed by the study team.</div>')
            continue
        for item in approved:
            page = f"<small>Protocol page {item.source_page}</small>" if item.source_page else ""
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

def render_chat_panel(guide: Optional[PatientGuide]) -> None:
    st.html('<div class="ctai-label">Study assistant</div><h3 style="margin:0 0 .5rem">Ask a question</h3>')

    box = st.container(height=470, border=False)
    with box:
        st.html(
            '<div class="ctai-chat-hello"><b>Hi! 👋</b> I can explain your study guide in everyday words. '
            'I only share what your study team has approved. I can\'t give medical advice or decide whether you can join.</div>'
        )
        for item in st.session_state.patient_history:
            with st.chat_message("user"):
                st.markdown(item["query"])
            with st.chat_message("assistant", avatar="🩺"):
                st.markdown(item["answer"].answer_text)

    st.caption("Try one of these:")
    cols = st.columns(2)
    for i, text in enumerate(SUGGESTIONS):
        if cols[i % 2].button(text, key=f"suggest_{i}"):
            ask(text)
            st.rerun()

    typed = st.chat_input("Type your question…", key="patient_chat_input")
    if typed:
        ask(typed)
        st.rerun()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render_patient_workspace() -> None:
    render_top_bar(PATIENT)

    name = st.session_state.get(NAME_KEY) or ""
    head, ctrl = st.columns([3, 1.4], vertical_alignment="bottom")
    with head:
        theme.page_head(
            f"Welcome, {name}" if name else "Welcome to your study guide",
            "Your guide on the left. Your assistant on the right.",
        )
    with ctrl:
        size = st.select_slider("Chat size", options=CHAT_SIZES, value="Balanced", key="chat_size")

    guide = get_guide()
    left, right = st.columns(_CHAT_RATIOS.get(size, _CHAT_RATIOS["Balanced"]), gap="large")
    with left:
        with st.container(key="guide_panel"):
            render_guide_panel(guide)
    with right:
        with st.container(key="chat_panel"):
            render_chat_panel(guide)
