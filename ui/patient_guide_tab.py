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
    reject_item,
    unapprove_item,
)
from models.patient_models import PatientAnswer
from models.plain_language_models import GuideItem, PatientGuide
from services.model_provider import provider_display_name
from ui import theme

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
    if item.rejected:
        return "❌ Rejected"
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


_SECTION_LABEL = {
    "inclusion_criteria": "Who can join",
    "exclusion_criteria": "Who cannot join",
    "visits": "Visits",
    "procedures": "Procedures",
    "safety_information": "Safety",
}


def _status_chip(item: GuideItem) -> str:
    if item.approved:
        return theme.chip("Approved", "teal")
    if item.rejected:
        return theme.chip("Rejected", "coral")
    if item.needs_review:
        return theme.chip("Needs review", "sun")
    return theme.chip("Waiting for your decision", "grey")


def _render_review_item(
    item: GuideItem,
    key: str,
    number: int = 0,
    total: int = 0,
    guide_complete: bool = True,
) -> None:
    with st.container(border=True):
        page = f"Protocol page {item.source_page}" if item.source_page else "Page not found"
        count = f"Statement {number} of {total} · " if number else ""
        st.html(
            '<div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">'
            f'<span class="ctai-label" style="margin:0">{count}{theme.e(_SECTION_LABEL.get(item.section, item.section))} · {theme.e(page)}</span>'
            f'{_status_chip(item)}</div>'
        )
        st.html(f'<div class="ctai-label">What the protocol says</div><div class="ctai-quote">{theme.e(item.original_text)}</div>')
        st.html(f'<div class="ctai-label">What patients would read</div><div class="ctai-plain">{theme.e(item.plain_text)}</div>')

        if item.review_reasons:
            for reason in item.review_reasons:
                st.html(f'<div class="ctai-check bad">⚠️ {theme.e(reason)}</div>')
        else:
            st.html('<div class="ctai-check ok">✔ Every number kept · ✔ No medical advice or eligibility wording</div>')

        st.html('<div class="ctai-label" style="margin-top:.6rem">Is this statement accurate?</div>')
        col_yes, col_no, _ = st.columns([1.3, 1.3, 2.4])

        with st.expander("Edit the wording", expanded=False):
            new_text = st.text_area(
                "Patient-friendly wording",
                value=item.plain_text,
                key=f"{key}_text",
                height=90,
                label_visibility="collapsed",
            )
            if st.button("Save edit", key=f"{key}_save"):
                warnings = edit_item(item, new_text)
                if warnings:
                    st.session_state[f"{key}_msg"] = ("warning", "Saved, but please check: " + " ".join(warnings))
                else:
                    st.session_state[f"{key}_msg"] = ("success", "Saved. Approve it to publish.")
                st.rerun()

        if item.approved:
            if col_yes.button("Withdraw approval", key=f"{key}_unapprove"):
                unapprove_item(item)
                st.rerun()
        else:
            approval_allowed = (
                guide_complete and item.analysis_complete is True
            )
            if col_yes.button(
                "✅ Yes, approve",
                key=f"{key}_approve",
                type="primary",
                disabled=not approval_allowed,
            ):
                if new_text.strip() != item.plain_text:
                    edit_item(item, new_text)
                if guide_complete and item.analysis_complete is True:
                    approve_item(item)
                else:
                    st.warning(
                        "Approval is disabled because analysis is incomplete "
                        "or its completeness could not be verified."
                    )
                st.rerun()
            if not item.rejected:
                with col_no.container(key=f"{key}_reject_style"):
                    if st.button("❌ No, reject", key=f"{key}_reject"):
                        reject_item(item)
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
        "and flags anything it cannot fix. For each statement, tell us if it is "
        "accurate. Patients only see what you approve."
    )

    guide = get_guide()
    if guide is None:
        if st.button("Generate patient statements", type="primary", key="guide_generate_btn"):
            try:
                _generate(pipeline_result)
            except Exception as exc:  # noqa: BLE001 — show any failure to the user
                st.error(f"Could not generate the guide: {exc}")
                return
            st.rerun()
        st.info(
            f"Uses the configured model provider: {provider_display_name()}. "
            "If a model request fails, the original wording is kept and "
            "flagged for review."
        )
        return

    completeness_verified = (
        guide.analysis_complete is True
        and all(item.analysis_complete is True for item in guide.all_items())
    )
    if not completeness_verified:
        st.warning(
            "This guide comes from incomplete or unverified protocol analysis. "
            "Approval and patient publication are disabled until a complete "
            "analysis is available. You may still withdraw existing approvals."
        )

    total = len(guide.all_items())
    flagged = len(guide.flagged_items())
    approved = guide.approved_count()
    rejected = guide.rejected_count()

    theme.metrics_row([
        ("Statements", total, False),
        ("Approved", f"{approved} / {total}", False),
        ("Need review", flagged, flagged > 0),
        ("Rejected", rejected, rejected > 0),
    ])
    theme.progress_bar(approved / total if total else 0)

    if guide.is_fully_approved():
        st.success("Every item is approved. The full guide is visible to patients.")
    elif approved:
        st.info("Patients can see the approved items. The rest stay hidden until approved.")
    else:
        st.info("Nothing is published yet. Approve items to make them visible to patients.")

    b1, b2, b3 = st.columns(3)
    if b1.button(
        "Approve all that passed checks",
        key="guide_approve_all_btn",
        disabled=not completeness_verified,
    ):
        if completeness_verified:
            approve_all_unflagged(guide)
        else:
            st.warning(
                "Approval is disabled because analysis is incomplete or "
                "its completeness could not be verified."
            )
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

    number = 0
    for s_index, section in enumerate(guide.sections()):
        items = [i for i in section.items if i.needs_review or not only_flagged]
        if only_flagged and not items:
            number += len(section.items)
            continue
        st.subheader(section.title)
        if section.unavailable_message:
            st.caption(section.unavailable_message)
            continue
        for i_index, item in enumerate(section.items):
            number += 1
            if item in items:
                _render_review_item(
                    item,
                    key=f"guide_{s_index}_{i_index}",
                    number=number,
                    total=total,
                    guide_complete=guide.analysis_complete is True,
                )


# ---------------------------------------------------------------------------
# Patient: approved guide only
# ---------------------------------------------------------------------------

def render_published_guide() -> None:
    """Show the approved parts of the guide to patients."""
    guide = get_guide()
    with st.expander("📖 Patient guide — approved by the study team", expanded=True):
        if guide is not None and guide.analysis_complete is not True:
            st.warning(
                "This protocol analysis is incomplete or could not be verified. "
                "Patient guide content and Q&A are unavailable until analysis "
                "is complete."
            )
            return
        if guide is None or guide.approved_count() == 0:
            st.caption(
                "The study team has not published a plain-language guide yet. "
                "You can still ask questions below."
            )
            return
        if not guide.is_fully_approved():
            st.caption("Some parts are still being reviewed by the study team.")
        st.markdown(guide_to_markdown(guide, approved_only=True, heading_level=3))
