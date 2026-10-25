"""Coordinator panels: critical-category coverage and patient summary editing."""

from __future__ import annotations

import sqlite3
from typing import Any, Mapping

import streamlit as st

from models.plain_language_models import PatientGuide
from services import authz, review_store
from services.coverage import build_coverage, override_block_reason
from services.summary_review import (
    change_diff,
    format_summary_markdown,
    generate_patient_summary,
    is_summary_approved,
    summary_approval_block_reason,
    validate_summary,
)

_STATUS_TEXT = {
    "covered": "✅ Covered",
    "missing": "❌ No statement",
    "unverified": "⚠️ Not verified",
}


def render_coverage_panel(
    pipeline_result: Mapping[str, Any], guide: PatientGuide, protocol_id: str
) -> None:
    """Pages and patient-facing categories, with gaps and N/A overrides."""
    coverage = build_coverage(pipeline_result, guide)
    pages = coverage["pages"]
    complete = pipeline_result.get("analysis_complete") is True
    overrides = review_store.load_coverage_overrides(protocol_id)

    with st.expander("Completeness check (pages and required topics)", expanded=bool(coverage["gaps"])):
        if pages:
            st.caption(
                f"Pages read from the PDF: {pages.get('extracted')} of "
                f"{pages.get('total')} · pages analyzed by the model: "
                f"{pages.get('analyzed')}. Pages with no analyzed content are "
                "administrative or unreadable."
            )
        st.caption(
            "Generating statements does not prove every topic was captured. "
            "Each topic below needs at least one approved, verified statement "
            "before publication."
        )
        rows = []
        for key, entry in coverage["categories"].items():
            note = "Not applicable (recorded)" if key in overrides else _STATUS_TEXT[entry["status"]]
            rows.append({
                "Topic": entry["label"],
                "Status": note,
                "Extracted": entry["extracted"],
                "Statements": entry.get("statements", 0),
                "Verified": entry.get("verified", 0),
                "Approved": entry.get("approved", 0),
                "Source pages": ", ".join(str(p) for p in entry["source_pages"]) or "–",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        for key, entry in coverage["categories"].items():
            if key in overrides:
                st.caption(f"{entry['label']}: marked not applicable — {overrides[key]['reason']}")
                if st.button("Remove override", key=f"cov_clear_{key}"):
                    review_store.clear_coverage_override(protocol_id, key)
                    st.rerun()
            elif entry["extracted"] == 0:
                reason = st.text_input(
                    f"{entry['label']}: reason this does not apply (after checking the protocol)",
                    key=f"cov_reason_{key}",
                )
                if st.button("Mark not applicable", key=f"cov_set_{key}"):
                    blocked = override_block_reason(coverage, key, reason, complete)
                    if blocked:
                        st.error(blocked)
                    else:
                        review_store.set_coverage_override(protocol_id, key, reason.strip())
                        st.rerun()


def render_summary_panel(guide: PatientGuide, protocol_id: str) -> None:
    """View, edit, save, review changes and approve the patient summary."""
    st.subheader("Patient summary")
    try:
        record = review_store.load_summary(protocol_id)
    except sqlite3.Error as exc:
        st.error(f"The summary could not be loaded: {exc}")
        return

    if record is None:
        st.info(
            "No patient summary has been generated yet. Go to **Step 2 — "
            "Summary** and click **Generate summary**; it will appear here "
            "for your review."
        )
        return

    approved = is_summary_approved(record)
    st.markdown("**Status:** " + ("✅ Approved" if approved else "📝 Not approved yet"))
    st.markdown(format_summary_markdown(record["draft_text"]))

    with st.expander("Edit summary", expanded=False):
        # Typed-but-unsaved text lives in a plain session key so it survives
        # reruns and leaving the page (widget state is dropped when the widget
        # is not drawn).
        unsaved_key = f"summary_unsaved_{protocol_id}"
        widget_key = f"summary_text_{protocol_id}_{record['updated_at']}"

        def _remember_unsaved() -> None:
            st.session_state[unsaved_key] = st.session_state[widget_key]

        edited = st.text_area(
            "Summary text",
            value=st.session_state.get(unsaved_key, record["draft_text"]),
            height=240,
            key=widget_key,
            on_change=_remember_unsaved,
        )
        if edited.strip() != record["draft_text"].strip():
            st.caption("You have unsaved changes. Click **Save draft** to keep them.")
        c1, c2 = st.columns(2)
        if c1.button("Save draft", key="summary_save_btn"):
            if not edited.strip():
                st.error("The summary cannot be empty.")
            elif review_store.save_summary_draft(protocol_id, edited.strip()):
                st.session_state.pop(unsaved_key, None)
                st.rerun()
            else:
                st.session_state.pop(unsaved_key, None)
                st.info("No changes to save.")
        if c2.button("Revert to AI original", key="summary_revert_btn"):
            st.session_state.pop(unsaved_key, None)
            if review_store.save_summary_draft(protocol_id, record["original_text"]):
                st.rerun()

    diff = change_diff(record["original_text"], record["draft_text"])
    with st.expander("Review changes from the AI original", expanded=False):
        if diff:
            st.code("\n".join(diff), language="diff")
        else:
            st.caption("No changes: the draft matches the AI original.")

    problems = validate_summary(record["draft_text"], guide)
    for problem in problems:
        st.warning(problem)

    block = summary_approval_block_reason(record, guide)
    if approved:
        if st.button("Withdraw summary approval", key="summary_unapprove_btn"):
            review_store.withdraw_summary_approval(protocol_id)
            st.rerun()
    else:
        unsaved = edited.strip() != record["draft_text"].strip()
        if st.button(
            "Approve summary", key="summary_approve_btn", type="primary",
            disabled=block is not None or unsaved,
        ):
            authz.require_coordinator()
            review_store.approve_summary(protocol_id)
            st.rerun()
        if unsaved:
            st.caption("Save your draft before approving, so the approved text is the text you reviewed.")
        elif block:
            st.caption("Cannot approve yet: " + block)

    with st.expander("Summary history"):
        for entry in review_store.summary_audit(protocol_id):
            st.caption(f"{entry['at']} · {entry['action']} · {entry['detail'] or ''}")


_REQUEST_KEY = "summary_generation_requested"


def render_summary_generation(pipeline_result: Mapping[str, Any], protocol_id: str) -> None:
    """
    Step 2: generate or regenerate the patient summary, only on request.

    The summary is written from the generated statements, so statement
    generation (the existing progressive job) runs first when it has not
    finished. An existing summary is never replaced unless the coordinator
    asks; replacing it withdraws its approval and is recorded in the history.
    """
    # Imported here: patient_guide_tab owns the statement-generation job.
    from ui.patient_guide_tab import _ensure_guide, get_guide

    st.subheader("Patient summary")
    try:
        record = review_store.load_summary(protocol_id)
    except sqlite3.Error as exc:
        st.error(f"The summary could not be loaded: {exc}")
        return

    requested = st.session_state.get(_REQUEST_KEY) == protocol_id
    if record is not None:
        status = "✅ Approved" if is_summary_approved(record) else "📝 Not approved yet"
        st.markdown(f"**Status:** {status}")
        st.markdown(format_summary_markdown(record["draft_text"]))
        st.caption("Edit and approve the summary in Step 3 — Review.")
    else:
        st.caption(
            "A plain-language overview written from the verified statements. "
            "Generate it here, then review and approve it in Step 3."
        )

    if not requested:
        confirmed = True
        if record is not None:
            confirmed = st.checkbox(
                "Replace the current summary (and withdraw its approval)",
                key="summary_regenerate_ack",
            )
        label = "Regenerate summary" if record is not None else "Generate summary"
        if st.button(label, key="summary_generate_btn", type="primary", disabled=not confirmed):
            st.session_state[_REQUEST_KEY] = protocol_id
            st.rerun()
        return

    guide = _ensure_guide(pipeline_result)
    if guide is None or not guide.generation_complete:
        st.caption("Statements are being generated first; the summary follows automatically.")
        return
    try:
        with st.spinner("Writing the summary…"):
            text, by = generate_patient_summary(guide)
        review_store.save_original_summary(protocol_id, text, by)
    except ValueError as exc:
        st.error(str(exc))
    finally:
        st.session_state.pop(_REQUEST_KEY, None)
    st.rerun()
