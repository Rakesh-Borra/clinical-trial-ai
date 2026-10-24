"""
Streamlit screens for the Plain-Language Patient Guide.

- render_guide_review_tab: coordinator generates the guide, reviews flagged
  items, edits wording, and approves items before publication.
- render_published_guide: patient-facing view showing approved items only.
- term_definition_answer: answers "What does X mean?" from the fixed glossary.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
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
    reverify_item,
    unapprove_item,
)
from models.patient_models import PatientAnswer
from models.plain_language_models import GuideItem, PatientGuide
from services import guide_jobs, review_store
from services.guide_review import (
    ALL,
    APPROVED,
    EDITED,
    FILTERS,
    NEEDS_REVERIFICATION,
    PENDING,
    REJECTED,
    edit_problems,
    matches_filter,
    patient_visible_guide,
    reconsider_item,
    review_state,
    state_counts,
)
from services.model_provider import provider_display_name
from ui import theme

GUIDE_KEY = "patient_guide"
GUIDE_SOURCE_KEY = "patient_guide_source"
GUIDE_JOB_KEY = "guide_job_id"
GUIDE_ERROR_KEY = "guide_generation_error"
GUIDE_FILTER_KEY = "guide_filter"

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
    state = review_state(item)
    if state == EDITED:
        return "✏️ Edited and approved"
    if state == APPROVED:
        return "✅ Approved"
    if state == REJECTED:
        return "❌ Rejected"
    if state == NEEDS_REVERIFICATION:
        return "🔁 Needs reverification"
    if item.needs_review:
        return "⚠️ Needs review"
    return "⏳ Waiting for approval"


def get_guide() -> Optional[PatientGuide]:
    return st.session_state.get(GUIDE_KEY)


def is_fully_approved_snapshot(guide: Any) -> bool:
    """Validate that a patient-facing guide contains only complete approvals."""
    if not isinstance(guide, PatientGuide) or guide.analysis_complete is not True:
        return False
    items = guide.all_items()
    return bool(items) and all(
        item.analysis_complete is True
        and item.approved
        and not item.rejected
        and item.verification == "verified"
        for item in items
    )


def clear_guide() -> None:
    """Forget the guide and any generation in progress or failed."""
    guide_jobs.jobs.discard(st.session_state.get(GUIDE_JOB_KEY))
    for key in (GUIDE_KEY, GUIDE_SOURCE_KEY, GUIDE_JOB_KEY, GUIDE_ERROR_KEY):
        st.session_state.pop(key, None)
    purge_item_widget_state()


_ITEM_WIDGET_KEY = re.compile(r"^guide_\d+_\d+_")


def purge_item_widget_state() -> None:
    """
    Forget every per-statement widget value.

    Widget keys are positional, so after a guide is regenerated or restored
    the old text boxes would otherwise keep showing (and could re-apply) text
    that belongs to a different guide.
    """
    for key in [k for k in st.session_state if _ITEM_WIDGET_KEY.match(str(k))]:
        del st.session_state[key]


def persist_review(action: str = "", detail: str = "") -> bool:
    """
    Save the working guide to SQLite, optionally logging a coordinator action.

    Returns False (after showing an error) when the database write fails, so
    the caller can avoid reporting a save that did not happen.
    """
    guide = get_guide()
    protocol_id = st.session_state.get("coordinator_protocol_id")
    if guide is None or not protocol_id:
        return True
    record = st.session_state.get("coordinator_protocols", {}).get(protocol_id, {})
    try:
        review_store.save_draft(
            protocol_id,
            guide,
            name=record.get("name", ""),
            uploaded_name=st.session_state.get("uploaded_name"),
            upload_fingerprint=st.session_state.get("uploaded_fingerprint"),
            pipeline=st.session_state.get("pipeline_result"),
        )
        if action:
            review_store.log_action(protocol_id, action, detail)
    except (sqlite3.Error, OSError) as exc:
        st.error(f"Your change could not be saved to the database: {exc}")
        return False
    return True


# ---------------------------------------------------------------------------
# Coordinator: generate, review, edit, approve
# ---------------------------------------------------------------------------

def analysis_fingerprint(pipeline_result: Mapping[str, Any]) -> str:
    """Identifies the analysis a guide was written from."""
    analysis = pipeline_result["analysis"]
    return hashlib.sha1(analysis.model_dump_json().encode()).hexdigest()[:16]


def _job_key(fingerprint: str) -> str:
    return f"{st.session_state.get('coordinator_protocol_id') or ''}:{fingerprint}"


def _start_generation(pipeline_result: Mapping[str, Any], fingerprint: str) -> None:
    """Start (or join) the background job for this protocol's analysis."""
    analysis = pipeline_result["analysis"]

    def runner(advance):
        return generate_patient_guide(analysis, on_item=lambda _item: advance())

    st.session_state[GUIDE_ERROR_KEY] = None
    st.session_state[GUIDE_JOB_KEY] = guide_jobs.jobs.submit(
        _job_key(fingerprint), runner, max(count_statements(analysis), 1)
    )


def _collect_generation(fingerprint: str) -> None:
    """Move a finished job's outcome into session state (main thread only)."""
    job_id = st.session_state.get(GUIDE_JOB_KEY)
    job = guide_jobs.jobs.get(job_id)
    if job is None:
        st.session_state[GUIDE_JOB_KEY] = None
        return
    if job.active:
        return
    st.session_state[GUIDE_JOB_KEY] = None
    guide_jobs.jobs.release(job_id)
    if job.status == guide_jobs.DISCARDED or job.key != _job_key(fingerprint):
        return
    if job.status == guide_jobs.FAILED:
        st.session_state[GUIDE_ERROR_KEY] = str(job.error) or job.error.__class__.__name__
        return
    purge_item_widget_state()
    st.session_state[GUIDE_KEY] = job.result
    st.session_state[GUIDE_SOURCE_KEY] = fingerprint
    st.session_state[GUIDE_ERROR_KEY] = None
    persist_review("guide_generated", "Patient statements generated.")


@st.fragment(run_every=2)
def _poll_generation() -> None:
    """Refresh progress; rerun the app once the job has finished."""
    job = guide_jobs.jobs.get(st.session_state.get(GUIDE_JOB_KEY))
    if job is None or not job.active:
        st.rerun()
    if job.status == guide_jobs.QUEUED or job.completed == 0:
        st.progress(0.0, text="Starting… 0% complete")
    else:
        st.progress(
            job.fraction,
            text=(
                f"{int(job.fraction * 100)}% complete — explained "
                f"{job.completed} of {job.total} protocol statements"
            ),
        )
    st.info(
        f"Writing patient-friendly statements with {provider_display_name()} and "
        "checking each one. This keeps running if you leave this page, and "
        "the results are kept when you return."
    )


def _ensure_guide(pipeline_result: Mapping[str, Any]) -> Optional[PatientGuide]:
    """
    The guide for this analysis, generating it automatically when missing.

    Returns None while generation is running or after it failed (the status or
    a Retry button is drawn here). An existing guide is reused as is.
    """
    fingerprint = analysis_fingerprint(pipeline_result)
    guide = get_guide()
    source = st.session_state.get(GUIDE_SOURCE_KEY)
    if guide is not None and source is not None and source != fingerprint:
        clear_guide()
        guide = None
    if guide is not None:
        return guide

    _collect_generation(fingerprint)
    guide = get_guide()
    if guide is not None:
        return guide

    if guide_jobs.jobs.get(st.session_state.get(GUIDE_JOB_KEY)) is not None:
        _poll_generation()
        return None

    error = st.session_state.get(GUIDE_ERROR_KEY)
    if error:
        st.error(
            f"Could not generate the patient statements: {error}\n\n"
            "Your uploaded protocol and analysis are kept."
        )
        if st.button("Retry generation", type="primary", key="guide_retry_btn"):
            _start_generation(pipeline_result, fingerprint)
            st.rerun()
        return None

    _start_generation(pipeline_result, fingerprint)
    st.rerun()
    return None


_SECTION_LABEL = {
    "inclusion_criteria": "Who can join",
    "exclusion_criteria": "Who cannot join",
    "visits": "Visits",
    "procedures": "Procedures",
    "safety_information": "Safety",
}


def _published_texts() -> set[str]:
    """Statement wording that patients can currently read for this protocol."""
    record = st.session_state.get("published_protocols", {}).get(
        st.session_state.get("coordinator_protocol_id")
    )
    guide = record.get("guide") if isinstance(record, dict) else None
    if not isinstance(guide, PatientGuide):
        return set()
    return {i.plain_text for i in guide.all_items()}


def _status_chip(item: GuideItem, published: Optional[set[str]] = None) -> str:
    state = review_state(item)
    if state in (APPROVED, EDITED) and published and item.plain_text in published:
        return theme.chip("Published", "lav")
    if state == NEEDS_REVERIFICATION:
        return theme.chip("Needs reverification", "sun")
    if state == EDITED:
        return theme.chip("Edited · approved", "teal")
    if state == APPROVED:
        return theme.chip("Approved", "teal")
    if state == REJECTED:
        return theme.chip("Rejected", "coral")
    if item.needs_review:
        return theme.chip("Needs review", "sun")
    return theme.chip("Pending · waiting for your decision", "grey")


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
            f'{_status_chip(item, _published_texts())}</div>'
        )
        st.html(f'<div class="ctai-label">What the protocol says</div><div class="ctai-quote">{theme.e(item.original_text)}</div>')
        st.html(f'<div class="ctai-label">What patients would read</div><div class="ctai-plain">{theme.e(item.plain_text)}</div>')

        if item.review_reasons:
            for reason in item.review_reasons:
                st.html(f'<div class="ctai-check bad">⚠️ {theme.e(reason)}</div>')
        else:
            st.html('<div class="ctai-check ok">✔ Every number kept · ✔ No medical advice or eligibility wording</div>')

        st.html('<div class="ctai-label" style="margin-top:.6rem">Is this statement accurate?</div>')
        col_yes, col_no, col_undo, _ = st.columns([1.3, 1.3, 1.3, 1.1])
        state = review_state(item)

        saved_text = item.plain_text
        unsaved_edit = False

        with st.expander(
            "Edit the wording",
            expanded=state == NEEDS_REVERIFICATION,
        ):
            new_text = st.text_area(
                "Patient-friendly wording",
                value=item.plain_text,
                key=f"{key}_text",
                height=90,
                label_visibility="collapsed",
            )
            unsaved_edit = new_text.strip() != saved_text
            if unsaved_edit:
                st.caption("You have unsaved changes. Click **Save edit** to keep them.")
            if st.button("Save edit", key=f"{key}_save"):
                if not new_text.strip():
                    st.session_state[f"{key}_msg"] = ("error", "The statement cannot be empty.")
                elif not unsaved_edit:
                    st.session_state[f"{key}_msg"] = ("info", "No changes to save.")
                else:
                    warnings = edit_item(item, new_text)
                    if persist_review("edited", f"{_SECTION_LABEL.get(item.section, item.section)}: {item.plain_text}"):
                        text = (
                            "Edit saved. Reverify it against the protocol, then "
                            "approve it again; it is not visible to patients."
                        )
                        if warnings:
                            text = "Edit saved, but please check: " + " ".join(warnings) + " " + text
                        st.session_state[f"{key}_msg"] = (
                            "warning" if warnings else "success",
                            text,
                        )
                st.rerun()

        if state == NEEDS_REVERIFICATION:
            st.caption(
                "Edited wording must be rechecked against the protocol excerpt "
                "above before it can be approved."
            )
            if st.button("🔁 Reverify against protocol", key=f"{key}_reverify"):
                problems = reverify_item(item)
                if persist_review("reverified" if not problems else "reverify_failed", item.plain_text):
                    if problems:
                        st.session_state[f"{key}_msg"] = (
                            "error",
                            "Reverification failed: " + " ".join(problems),
                        )
                    else:
                        st.session_state[f"{key}_msg"] = (
                            "success",
                            "Reverified against the protocol. You can now approve it.",
                        )
                st.rerun()

        edit_issues = edit_problems(item)
        if state in (APPROVED, EDITED):
            if col_yes.button("Withdraw approval", key=f"{key}_unapprove"):
                unapprove_item(item)
                if persist_review("approval_withdrawn", item.plain_text):
                    st.session_state[f"{key}_msg"] = ("info", "Approval withdrawn.")
                st.rerun()
        else:
            approval_allowed = (
                guide_complete
                and item.analysis_complete is True
                and item.verification == "verified"
                and not edit_issues
                and not unsaved_edit
            )
            if col_yes.button(
                "✅ Yes, approve",
                key=f"{key}_approve",
                type="primary",
                disabled=not approval_allowed,
            ):
                if unsaved_edit:
                    st.session_state[f"{key}_msg"] = (
                        "error",
                        "Save your edit first, then reverify and approve it.",
                    )
                elif guide_complete and item.analysis_complete is True:
                    try:
                        approve_item(item)
                    except ValueError as exc:
                        st.session_state[f"{key}_msg"] = ("error", str(exc))
                    else:
                        if persist_review("approved", item.plain_text):
                            st.session_state[f"{key}_msg"] = (
                                "success",
                                "Approved. It will reach patients when you publish.",
                            )
                else:
                    st.session_state[f"{key}_msg"] = (
                        "error",
                        "Approval is disabled because analysis is incomplete "
                        "or its completeness could not be verified.",
                    )
                st.rerun()
        if state == REJECTED:
            if col_undo.button("↩ Reconsider", key=f"{key}_reconsider"):
                reconsider_item(item)
                if persist_review("reconsidered", item.plain_text):
                    st.session_state[f"{key}_msg"] = ("info", "Statement returned to Pending.")
                st.rerun()
        else:
            with col_no.container(key=f"{key}_reject_style"):
                if st.button("❌ No, reject", key=f"{key}_reject"):
                    reject_item(item)
                    if persist_review("rejected", item.plain_text):
                        st.session_state[f"{key}_msg"] = (
                            "success",
                            "Rejected. It will never be shown to patients.",
                        )
                    st.rerun()
        if edit_issues and state not in (APPROVED, EDITED):
            st.caption(
                "This edit has not passed verification, so it cannot be "
                "approved yet: " + " ".join(edit_issues)
            )

        message = st.session_state.pop(f"{key}_msg", None)
        if message:
            getattr(st, message[0])(message[1])

        if item.terms_explained:
            st.caption(
                "Words explained for patients: "
                + ", ".join(display_term(t.term) for t in item.terms_explained)
            )


def _render_filters(counts: dict[str, int]) -> str:
    """Filter buttons with live counts; returns the selected filter key."""
    selected = st.session_state.get(GUIDE_FILTER_KEY, ALL)
    if selected not in dict(FILTERS):
        selected = ALL
    with st.container(key="guide_filters", horizontal=True, gap="small"):
        for filter_key, label in FILTERS:
            if st.button(
                f"{label} ({counts[filter_key]})",
                key=f"guide_filter_{filter_key}",
                type="primary" if filter_key == selected else "secondary",
            ):
                st.session_state[GUIDE_FILTER_KEY] = filter_key
                st.rerun()
    return selected


def render_guide_review_tab(pipeline_result: Mapping[str, Any]) -> None:
    """Coordinator screen: statements are generated on entry, then reviewed."""
    st.caption(
        "The AI rewrites each protocol statement in plain language, checks "
        "that every number is kept and that nothing sounds like medical "
        "advice or an eligibility decision, retries once if a check fails, "
        "and flags anything it cannot fix. Give every statement a decision: "
        "approve it, edit and approve it, or reject it. Rejected statements "
        "stay here for audit but are never shown to patients."
    )

    guide = _ensure_guide(pipeline_result)
    if guide is None:
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

    counts = state_counts(guide)
    total = counts[ALL]
    pending = counts[PENDING]
    accepted = counts[APPROVED] + counts[EDITED]
    rejected = counts[REJECTED]

    theme.metrics_row([
        ("Statements", total, False),
        ("Approved", f"{accepted} / {total}", False),
        ("Pending", pending, pending > 0),
        ("Rejected", rejected, rejected > 0),
    ])
    theme.progress_bar((total - pending) / total if total else 0)

    if pending:
        st.info(
            f"{pending} statement(s) still need your decision. Publishing "
            "waits until every statement is approved, edited and approved, or rejected."
        )
    elif accepted:
        st.success(
            f"Every statement has a decision: {accepted} to publish, "
            f"{rejected} rejected. Rejected statements are never shown to patients."
        )
    else:
        st.warning("Every statement is rejected. Approve at least one to publish.")

    b1, b2, b3 = st.columns(3)
    if b1.button(
        "Approve all that passed checks",
        key="guide_approve_all_btn",
        disabled=not completeness_verified,
    ):
        if completeness_verified:
            approved_now = approve_all_unflagged(guide)
            persist_review("approved_all_checked", f"{approved_now} statement(s)")
        else:
            st.warning(
                "Approval is disabled because analysis is incomplete or "
                "its completeness could not be verified."
            )
        st.rerun()
    b2.download_button(
        "Download guide (Markdown)",
        data=guide_to_markdown(patient_visible_guide(guide)),
        file_name="patient_guide.md",
        mime="text/markdown",
        key="guide_download_btn",
    )
    if b3.button("Regenerate guide", key="guide_regenerate_btn"):
        clear_guide()
        st.rerun()

    selected = _render_filters(counts)
    if counts[selected] == 0 and selected != ALL:
        st.caption("No statements match this filter.")

    number = 0
    for s_index, section in enumerate(guide.sections()):
        visible = [i for i in section.items if matches_filter(i, selected)]
        if not visible and section.items:
            number += len(section.items)
            continue
        if selected != ALL and not section.items:
            continue
        st.subheader(section.title)
        if section.unavailable_message:
            st.caption(section.unavailable_message)
            continue
        for i_index, item in enumerate(section.items):
            number += 1
            if matches_filter(item, selected):
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
    """Show only the selected protocol's successfully published snapshot."""
    selected_id = st.session_state.get("patient_protocol_select")
    record = st.session_state.get("published_protocols", {}).get(selected_id)
    guide = record.get("guide") if isinstance(record, dict) else None
    with st.expander("📖 Patient guide — approved by the study team", expanded=True):
        if not is_fully_approved_snapshot(guide):
            st.caption("No published clinical trial protocol is selected.")
            return
        st.markdown(guide_to_markdown(guide, approved_only=True, heading_level=3))
