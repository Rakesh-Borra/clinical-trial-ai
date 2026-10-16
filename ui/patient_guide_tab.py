"""
Streamlit screens for the Plain-Language Patient Guide.

- render_guide_review_tab: coordinator generates the guide, reviews flagged
  items, edits wording, and approves items before publication.
- render_published_guide: patient-facing view showing approved items only.
- term_definition_answer: answers "What does X mean?" from the fixed glossary.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import streamlit as st

from agents.plain_language_agent import (
    approve_all_unflagged,
    approve_item,
    count_statements,
    display_term,
    edit_item,
    explain_term_query,
    generate_patient_guide,
    guide_to_markdown,
    unapprove_item,
)
from models.patient_models import PatientAnswer
from models.plain_language_models import GuideItem, PatientGuide

GUIDE_KEY = "patient_guide"

_TERM_DISCLAIMER = (
    "This is a general definition, not medical advice. Ask the study team "
    "what it means for you."
)


# ---------------------------------------------------------------------------
# Helpers that do not draw anything (easy to test)
# ---------------------------------------------------------------------------

def term_definition_answer(query: str) -> Optional[PatientAnswer]:
    """Build a PatientAnswer for definition questions, or None."""
    terms = explain_term_query(query)
    if not terms:
        return None
    lines = ["Here is what that means in everyday words:", ""]
    for term in terms:
        lines.append(f"- **{display_term(term.term)}**: {term.explanation}")
    return PatientAnswer(
        answer_text="\n".join(lines),
        intent="term_definition",
        confidence="high",
        limitations=["Definitions come from a fixed glossary of common trial terms."],
        disclaimers=[_TERM_DISCLAIMER],
        audience="patient",
    )


def item_status(item: GuideItem) -> str:
    """Short status label for the review screen."""
    if item.approved:
        return "✅ Approved"
    if item.needs_review:
        return "⚠️ Needs review"
    return "⏳ Waiting for approval"


def get_guide() -> Optional[PatientGuide]:
    return st.session_state.get(GUIDE_KEY)


def clear_guide() -> None:
    st.session_state.pop(GUIDE_KEY, None)


# ---------------------------------------------------------------------------
# Coordinator: generate, review, edit, approve
# ---------------------------------------------------------------------------

def _generate(pipeline_result: Mapping[str, Any]) -> None:
    analysis = pipeline_result["analysis"]
    total = max(count_statements(analysis), 1)
    done = {"n": 0}
    progress = st.progress(0, text="Writing plain-language explanations…")

    def on_item(_item: GuideItem) -> None:
        done["n"] += 1
        progress.progress(
            min(done["n"] / total, 1.0),
            text=f"Explained {done['n']} of {total} protocol statements…",
        )

    st.session_state[GUIDE_KEY] = generate_patient_guide(analysis, on_item=on_item)
    progress.empty()


def _render_review_item(item: GuideItem, key: str) -> None:
    with st.container(border=True):
        top_left, top_right = st.columns([4, 1])
        page = f" · protocol page {item.source_page}" if item.source_page else ""
        top_left.caption(f"Original protocol text{page}")
        top_left.markdown(f"> {item.original_text}")
        top_right.markdown(f"**{item_status(item)}**")
        top_right.caption(f"Source: {item.generated_by}")

        for reason in item.review_reasons:
            st.warning(reason)

        new_text = st.text_area(
            "Patient-friendly wording",
            value=item.plain_text,
            key=f"{key}_text",
            height=90,
        )

        col_save, col_approve, _ = st.columns([1, 1, 3])
        if col_save.button("Save edit", key=f"{key}_save"):
            warnings = edit_item(item, new_text)
            if warnings:
                st.session_state[f"{key}_msg"] = ("warning", "Saved, but please check: " + " ".join(warnings))
            else:
                st.session_state[f"{key}_msg"] = ("success", "Saved. Approve it to publish.")
            st.rerun()

        if item.approved:
            if col_approve.button("Unapprove", key=f"{key}_unapprove"):
                unapprove_item(item)
                st.rerun()
        else:
            if col_approve.button("Approve", key=f"{key}_approve", type="primary"):
                if new_text.strip() != item.plain_text:
                    edit_item(item, new_text)
                approve_item(item)
                st.rerun()

        message = st.session_state.pop(f"{key}_msg", None)
        if message:
            getattr(st, message[0])(message[1])

        if item.terms_explained:
            st.caption(
                "Words explained for patients: "
                + ", ".join(display_term(t.term) for t in item.terms_explained)
            )


def render_guide_review_tab(pipeline_result: Mapping[str, Any]) -> None:
    """Coordinator screen: generate the guide, review it, approve it."""
    st.caption(
        "The AI rewrites each protocol statement in plain language, checks "
        "that every number is kept and that nothing sounds like medical "
        "advice or an eligibility decision, retries once if a check fails, "
        "and flags anything it cannot fix. Patients only see items you approve."
    )

    guide = get_guide()
    if guide is None:
        if st.button("Generate patient guide", type="primary", key="guide_generate_btn"):
            try:
                _generate(pipeline_result)
            except Exception as exc:  # noqa: BLE001 — show any failure to the user
                st.error(f"Could not generate the guide: {exc}")
                return
            st.rerun()
        st.info(
            "Uses the local Ollama model. If Ollama is not running, every item "
            "keeps the original protocol wording and is flagged for review."
        )
        return

    total = len(guide.all_items())
    flagged = len(guide.flagged_items())
    approved = guide.approved_count()

    m1, m2, m3 = st.columns(3)
    m1.metric("Statements", total)
    m2.metric("Need review", flagged)
    m3.metric("Approved", f"{approved} / {total}")

    if guide.is_fully_approved():
        st.success("Every item is approved. The full guide is visible to patients.")
    elif approved:
        st.info("Patients can see the approved items. The rest stay hidden until approved.")
    else:
        st.info("Nothing is published yet. Approve items to make them visible to patients.")

    b1, b2, b3 = st.columns(3)
    if b1.button("Approve all that passed checks", key="guide_approve_all_btn"):
        approve_all_unflagged(guide)
        st.rerun()
    b2.download_button(
        "Download guide (Markdown)",
        data=guide_to_markdown(guide),
        file_name="patient_guide.md",
        mime="text/markdown",
        key="guide_download_btn",
    )
    if b3.button("Regenerate guide", key="guide_regenerate_btn"):
        clear_guide()
        st.rerun()

    only_flagged = st.toggle("Show only items that need review", key="guide_only_flagged")
    if only_flagged and flagged == 0:
        st.success("No items need review: every rewrite passed the automatic checks.")

    for s_index, section in enumerate(guide.sections()):
        items = [i for i in section.items if i.needs_review or not only_flagged]
        if only_flagged and not items:
            continue
        st.subheader(section.title)
        if section.unavailable_message:
            st.caption(section.unavailable_message)
            continue
        for i_index, item in enumerate(section.items):
            if item in items:
                _render_review_item(item, key=f"guide_{s_index}_{i_index}")


# ---------------------------------------------------------------------------
# Patient: approved guide only
# ---------------------------------------------------------------------------

def render_published_guide() -> None:
    """Show the approved parts of the guide to patients."""
    guide = get_guide()
    with st.expander("📖 Patient guide — approved by the study team", expanded=True):
        if guide is None or guide.approved_count() == 0:
            st.caption(
                "The study team has not published a plain-language guide yet. "
                "You can still ask questions below."
            )
            return
        if not guide.is_fully_approved():
            st.caption("Some parts are still being reviewed by the study team.")
        st.markdown(guide_to_markdown(guide, approved_only=True, heading_level=3))
