"""Coordinator panels: critical-category coverage and patient summary editing."""

from __future__ import annotations

import hashlib
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


def _text_digest(text: str) -> str:
    """Ties the 'I have read it' tick to this exact text, so any edit resets it."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def _discard_summary_edit(widget_key: str, unsaved_key: str, saved: str) -> None:
    """Cancel Edits: back to the last saved draft (runs before widgets are drawn)."""
    st.session_state[widget_key] = saved
    st.session_state.pop(unsaved_key, None)


def render_summary_panel(guide: PatientGuide, protocol_id: str) -> None:
    """Section 3: view, edit, save and approve the patient summary (collapsed)."""
    try:
        record = review_store.load_summary(protocol_id)
    except sqlite3.Error as exc:
        st.error(f"The summary could not be loaded: {exc}")
        return

    unsaved = False
    approved = False
    reviewed_key = ""
    with st.expander("Patient Summary", expanded=False):
        if record is None:
            st.info(
                "No patient summary has been generated yet. Go to **Step 2 — "
                "Summary** and click **Generate summary**; it will appear here "
                "for your review."
            )
            return

        approved = is_summary_approved(record)
        saved_note = st.session_state.pop(f"summary_saved_{protocol_id}", None)
        if saved_note:
            st.success(saved_note)
        st.markdown("**Status:** " + ("✅ Approved" if approved else "📝 Not approved yet"))

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
            height=260,
            key=widget_key,
            on_change=_remember_unsaved,
            label_visibility="collapsed",
        )
        unsaved = edited.strip() != record["draft_text"].strip()
        if unsaved:
            st.caption("You have unsaved changes. Save Changes to keep them, or Cancel Edits.")
        save_col, cancel_col, revert_col = st.columns(3)
        if save_col.button("Save Changes", key="summary_save_btn", disabled=not unsaved):
            authz.require_coordinator()
            if not edited.strip():
                st.error("The summary cannot be empty.")
            elif review_store.save_summary_draft(protocol_id, edited.strip()):
                st.session_state.pop(unsaved_key, None)
                st.session_state[f"summary_saved_{protocol_id}"] = (
                    "Summary saved. It needs to be approved again before publishing."
                )
                st.rerun()
        cancel_col.button(
            "Cancel Edits",
            key="summary_cancel_btn",
            disabled=not unsaved,
            on_click=_discard_summary_edit,
            args=(widget_key, unsaved_key, record["draft_text"]),
        )
        if revert_col.button("Revert to AI original", key="summary_revert_btn"):
            st.session_state.pop(unsaved_key, None)
            if review_store.save_summary_draft(protocol_id, record["original_text"]):
                st.rerun()

        # Claims in the saved draft that the approved statements do not support.
        for problem in validate_summary(record["draft_text"], guide):
            st.warning(problem)

        reviewed_key = f"summary_reviewed_{protocol_id}_{_text_digest(record['draft_text'])}"
        if not approved and not unsaved:
            st.checkbox(
                "I have read this summary and it is ready to approve",
                key=reviewed_key,
            )

        changes_tab, history_tab = st.tabs(["Changes from the AI original", "Summary history"])
        with changes_tab:
            diff = change_diff(record["original_text"], record["draft_text"])
            if diff:
                st.code("\n".join(diff), language="diff")
            else:
                st.caption("No changes: the draft matches the AI original.")
        with history_tab:
            for entry in review_store.summary_audit(protocol_id):
                st.caption(f"{entry['at']} · {entry['action']} · {entry['detail'] or ''}")


    block = summary_approval_block_reason(record, guide)
    if approved:
        st.success("Summary Approved")
        if st.button("Withdraw summary approval", key="summary_unapprove_btn"):
            review_store.withdraw_summary_approval(protocol_id)
            st.rerun()
        return
    reviewed = bool(st.session_state.get(reviewed_key))
    if st.button(
        "Approve Summary", key="summary_approve_btn", type="primary",
        disabled=block is not None or unsaved or not reviewed,
    ):
        authz.require_coordinator()
        review_store.approve_summary(protocol_id)
        st.rerun()
    if unsaved:
        st.caption("Save your changes before approving, so the approved text is the text you reviewed.")
    elif block:
        st.caption("Cannot approve yet: " + block)
    elif not reviewed:
        st.caption("Expand Patient Summary, read it, and tick the review box to enable approval.")


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
