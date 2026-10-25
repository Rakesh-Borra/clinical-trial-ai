"""
Streamlit screens for the Plain-Language Patient Guide.

- render_guide_review_tab: coordinator generates the guide, reviews flagged
  items, edits wording, and approves items before publication.
- render_published_guide: patient-facing view showing approved items only.
- term_definition_answer: answers "What does X mean?" from the fixed glossary.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import time
from typing import Any, Mapping, Optional

import streamlit as st

from agents.plain_language_agent import (
    approve_all_unflagged,
    approve_all_with_override,
    approval_block_reason,
    approve_item,
    item_cleared,
    count_statements,
    display_term,
    edit_item,
    ensure_item_ids,
    explain_term_query,
    generate_patient_guide,
    merge_generated,
    new_guide_shell,
    guide_to_markdown,
    reject_item,
    recheck_sources,
    reverify_item,
    unapprove_item,
    verification_warnings,
)
from models.patient_models import PatientAnswer
from models.plain_language_models import (
    GuideItem,
    PatientGuide,
    as_patient_guide,
    page_label,
)
from services import authz, benchmark_log, guide_jobs, review_store, version_store
from services.guide_review import (
    ALL,
    APPROVED,
    EDITED,
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
from services.source_inspector import inspect_source
from services.source_verifier import verify_saved_pages
from services.manual_statements import SECTION_ATTR
from ui import theme
from ui.add_statement import render_add_statement

GUIDE_KEY = "patient_guide"
GUIDE_SOURCE_KEY = "patient_guide_source"
GUIDE_JOB_KEY = "guide_job_id"
GUIDE_ERROR_KEY = "guide_generation_error"
GUIDE_AUTOSTART_KEY = "guide_autostart_fingerprint"
MANUAL_KEEP_KEY = "guide_manual_keep"
FILTER_KEY = "guide_review_filter"
APPROVE_ALL_CONFIRM_KEY = "guide_approve_all_pending"

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
    return "⏳ Waiting for your decision"


def get_guide() -> Optional[PatientGuide]:
    """The working guide, re-hydrated onto the current PatientGuide class if needed."""
    raw = st.session_state.get(GUIDE_KEY)
    guide = as_patient_guide(raw)
    if guide is not None and guide is not raw:
        st.session_state[GUIDE_KEY] = guide
    return guide


def final_validation_reason(guide: Any) -> Optional[str]:
    """The specific requirement a patient-facing guide fails, or None when it passes."""
    guide = as_patient_guide(guide)
    if guide is None:
        return "The patient guide could not be read."
    if guide.analysis_complete is not True:
        return "The analysis behind this guide is not marked complete."
    if not guide.generation_complete:
        return "Statement generation has not finished (or some statements failed)."
    items = guide.all_items()
    if not items:
        return "No approved statements would be published."
    for item in items:
        label = f"“{item.plain_text[:60]}” (page {item.source_page or '?'})"
        if item.analysis_complete is not True:
            return f"Statement {label} did not complete analysis verification."
        if item.rejected or not item.approved:
            return f"Statement {label} is not approved."
        if not item_cleared(item):
            return f"Statement {label} has unresolved verification warnings."
    # Approve on a flagged statement records the override; no second confirm.
    return None


def is_fully_approved_snapshot(guide: Any) -> bool:
    """Validate that a patient-facing guide contains only complete approvals."""
    return final_validation_reason(guide) is None


def clear_guide() -> None:
    """Forget the guide and any generation in progress or failed."""
    guide_jobs.jobs.discard(st.session_state.get(GUIDE_JOB_KEY))
    for key in (GUIDE_KEY, GUIDE_SOURCE_KEY, GUIDE_JOB_KEY, GUIDE_ERROR_KEY, GUIDE_AUTOSTART_KEY):
        st.session_state.pop(key, None)
    purge_item_widget_state()


_ITEM_WIDGET_KEY = re.compile(r"^guide_[0-9a-f]{12}(?:-\d+)?_")


def purge_item_widget_state() -> None:
    """
    Forget every per-statement widget value.

    Widget keys use the statement id, so after a guide is regenerated or
    restored the old text boxes would otherwise keep showing (and could
    re-apply) text that belongs to a different guide.
    """
    for key in [k for k in st.session_state if _ITEM_WIDGET_KEY.match(str(k))]:
        del st.session_state[key]


def _store_original_pdf(protocol_id: str) -> None:
    """Keep the uploaded PDF with its version (once, with an integrity hash)."""
    path = st.session_state.get("temp_pdf_path")
    version = version_store.get_version(protocol_id)
    if (
        not path
        or not os.path.isfile(path)
        or version is None
        or version["pdf_status"] == version_store.PDF_STORED
    ):
        return
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        version_store.attach_pdf(protocol_id, data, st.session_state.get("uploaded_name") or "")
    except (version_store.VersionError, OSError) as exc:
        st.warning(f"The original PDF could not be stored with this version: {exc}")


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
            study_id=st.session_state.get("coord_study_id"),
            version_label=st.session_state.get("coord_version_label"),
        )
        _store_original_pdf(protocol_id)
        if action:
            review_store.log_action(protocol_id, action, detail)
    except version_store.VersionError as exc:
        st.error(f"Your change was not saved: {exc}")
        return False
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


def _store_id() -> str:
    return st.session_state.get("coordinator_protocol_id") or "_"


def _start_generation(
    pipeline_result: Mapping[str, Any],
    fingerprint: str,
    retry_failed: bool = False,
) -> None:
    """
    Start (or join) the background job for this protocol's analysis.

    Each statement is saved to SQLite the moment it is generated. The job
    skips statements that already exist, so a restarted or retried job only
    does the missing work (and, for a retry, the failed work).
    """
    analysis = pipeline_result["analysis"]
    verification = pipeline_result.get("verification")
    store_id = _store_id()
    label = benchmark_log.label_for(st.session_state.get("temp_pdf_path") or "")
    guide = get_guide()
    known = {i.item_id: i for i in guide.all_items() if i.item_id} if guide else {}

    def runner(advance):
        stored = review_store.load_generated_items(store_id, fingerprint)
        reuse = {**stored, **known}
        if retry_failed:
            reuse = {k: v for k, v in reuse.items() if not v.generation_failed}

        def on_item(item):
            if item.item_id not in reuse:
                review_store.save_generated_item(store_id, fingerprint, item)
            advance()

        started = time.perf_counter()
        try:
            guide_result = generate_patient_guide(
                analysis, on_item=on_item, verification=verification, reuse=reuse,
                source_pages=pipeline_result.get("source_pages"),
            )
        except Exception as exc:
            benchmark_log.record(
                "statements", label, success=False,
                seconds=round(time.perf_counter() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        items = guide_result.all_items()
        benchmark_log.record(
            "statements", label,
            success=not any(i.generation_failed for i in items),
            seconds=round(time.perf_counter() - started, 3),
            statements=len(items),
            reused=sum(1 for i in items if i.item_id in reuse),
            failed=sum(1 for i in items if i.generation_failed),
            warnings=sum(1 for i in items if verification_warnings(i)),
        )
        return guide_result

    st.session_state[GUIDE_ERROR_KEY] = None
    st.session_state[GUIDE_AUTOSTART_KEY] = fingerprint
    st.session_state[GUIDE_JOB_KEY] = guide_jobs.jobs.submit(
        _job_key(fingerprint), runner, max(count_statements(analysis), 1)
    )


def _collect_generation(fingerprint: str) -> None:
    """Record a finished job's outcome (main thread only). Items arrive via the store."""
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


def _stored_items(fingerprint: str) -> dict:
    try:
        return review_store.load_generated_items(_store_id(), fingerprint)
    except (sqlite3.Error, OSError):
        return {}


def _sync_generated(pipeline_result: Mapping[str, Any], fingerprint: str) -> Optional[PatientGuide]:
    """
    Add statements generated since the last look to the session guide.

    Existing statements keep their object, so decisions made while generation
    continued are never overwritten. Saves the draft when anything arrived.
    """
    analysis = pipeline_result["analysis"]
    guide = get_guide()
    if guide is not None and guide.generation_complete:
        return guide
    stored = _stored_items(fingerprint)
    if guide is None:
        if not stored and count_statements(analysis) > 0:
            return None
        guide = new_guide_shell(analysis, count_statements(analysis), False)
        st.session_state[GUIDE_KEY] = guide
        st.session_state[GUIDE_SOURCE_KEY] = fingerprint
        for kept in st.session_state.pop(MANUAL_KEEP_KEY, []):
            getattr(guide, SECTION_ATTR[kept.section]).items.append(kept)
    was_complete = guide.generation_complete
    if merge_generated(guide, analysis, stored):
        done = guide.generation_complete and not was_complete
        persist_review(
            "guide_generated" if done else "",
            "Patient statements generated." if done else "",
        )
    return guide


@st.fragment(run_every=2)
def _poll_generation(fingerprint: str) -> None:
    """Show live progress; do a full rerun when statements arrive or the job ends."""
    job = guide_jobs.jobs.get(st.session_state.get(GUIDE_JOB_KEY))
    guide = get_guide()
    shown = {i.item_id for i in guide.all_items()} if guide else set()
    stored = _stored_items(fingerprint)
    if job is None or not job.active or not set(stored) <= shown:
        st.rerun()
    total = job.total
    done = min(len(stored), total)
    failed = sum(1 for i in stored.values() if i.generation_failed)
    fraction = done / total if total else 0.0
    suffix = f" · {failed} failed" if failed else ""
    st.progress(fraction, text=f"{done} / {total} generated — {int(fraction * 100)}%{suffix}")
    st.info(
        f"Writing patient-friendly statements with {provider_display_name()} and "
        "checking each one. Finished statements appear below and can be "
        "approved, edited or rejected now. Generation keeps running if you "
        "leave this page, and finished statements are kept."
    )


def _ensure_guide(pipeline_result: Mapping[str, Any]) -> Optional[PatientGuide]:
    """
    The guide for this analysis, which may still be generating.

    Statements appear as they are finished. Returns None only before the first
    statement exists. A complete saved guide is reused as is; an interrupted
    one resumes once automatically, then offers a retry for failed work.
    """
    fingerprint = analysis_fingerprint(pipeline_result)
    guide = get_guide()
    source = st.session_state.get(GUIDE_SOURCE_KEY)
    if guide is not None and source is not None and source != fingerprint:
        clear_guide()
        guide = None
    if guide is not None and ensure_item_ids(guide):
        persist_review()

    _collect_generation(fingerprint)
    guide = _sync_generated(pipeline_result, fingerprint)
    if guide is not None and guide.generation_complete:
        return guide

    job_active = guide_jobs.jobs.get(st.session_state.get(GUIDE_JOB_KEY)) is not None
    if job_active:
        _poll_generation(fingerprint)
        return guide

    error = st.session_state.get(GUIDE_ERROR_KEY)
    failed = guide is not None and any(
        i.generation_failed and not i.rejected for i in guide.all_items()
    )
    if error:
        st.error(
            f"Could not generate the patient statements: {error}\n\n"
            "Your uploaded protocol, analysis and finished statements are kept."
        )
        if st.button("Retry generation", type="primary", key="guide_retry_btn"):
            _start_generation(pipeline_result, fingerprint, retry_failed=True)
            st.rerun()
        return guide
    if failed:
        count = sum(1 for i in guide.all_items() if i.generation_failed and not i.rejected)
        st.warning(
            f"{count} statement(s) could not be generated. Everything else is "
            "available to review. Retrying only redoes the failed ones."
        )
        if st.button("Retry failed statements", type="primary", key="guide_retry_btn"):
            _start_generation(pipeline_result, fingerprint, retry_failed=True)
            st.rerun()
        return guide
    if st.session_state.get(GUIDE_AUTOSTART_KEY) != fingerprint:
        _start_generation(pipeline_result, fingerprint)
        st.rerun()
    st.warning("Statement generation stopped before finishing.")
    if st.button("Resume generation", type="primary", key="guide_retry_btn"):
        _start_generation(pipeline_result, fingerprint)
        st.rerun()
    return guide


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
    if state == EDITED:
        return theme.chip("Edited · approved", "teal")
    if state == APPROVED:
        return theme.chip("Approved", "teal")
    if state == REJECTED:
        return theme.chip("Rejected", "coral")
    return theme.chip("Pending · waiting for your decision", "grey")


def _restore_text(widget_key: str, saved_text: str) -> None:
    """Discard an unsaved edit (runs as a callback, before the widget is drawn)."""
    st.session_state[widget_key] = saved_text


def _override_audit_detail(item: GuideItem) -> str:
    """Facts for the audit log — no fabricated approval reason."""
    user = authz.current_user() or {}
    actor = user.get("email") or user.get("user_id") or "coordinator"
    warnings = item.override_warnings or verification_warnings(item)
    return (
        f"actor={actor} | item_id={item.item_id} | at={item.override_at or ''} "
        f"| warnings={len(warnings)} | status=flagged_approved | {item.plain_text}"
    )


def _mark_override_confirmed(guide: PatientGuide, item: GuideItem) -> None:
    """Approve is the confirmation; keep overrides_confirmed in sync for snapshots."""
    if item.item_id and item.item_id not in guide.overrides_confirmed:
        guide.overrides_confirmed = sorted(
            set(guide.overrides_confirmed) | {item.item_id}
        )


def _clear_override_confirmation(guide: PatientGuide, item: GuideItem) -> None:
    if item.item_id in guide.overrides_confirmed:
        guide.overrides_confirmed = [
            i for i in guide.overrides_confirmed if i != item.item_id
        ]


def _render_review_item(
    item: GuideItem,
    key: str,
    number: int = 0,
    total: int = 0,
) -> None:
    with st.container(border=True):
        page = f"Protocol {page_label(item)}" if item.source_page else "Page not found"
        count = f"Statement {number} of {total} · " if number else ""
        count += "Added by coordinator · " if item.added_by_coordinator else ""
        st.html(
            '<div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">'
            f'<span class="ctai-label" style="margin:0">{count}{theme.e(_SECTION_LABEL.get(item.section, item.section))} · {theme.e(page)}</span>'
            f'{_status_chip(item, _published_texts())}</div>'
        )
        if not item.added_by_coordinator:
            st.html(f'<div class="ctai-label">What the protocol says</div><div class="ctai-quote">{theme.e(item.original_text)}</div>')
        st.html(f'<div class="ctai-label">What patients would read</div><div class="ctai-plain">{theme.e(item.plain_text)}</div>')

        state = review_state(item)
        warnings = verification_warnings(item)
        flagged = bool(warnings)
        decided = state in (APPROVED, EDITED)

        if flagged and state == PENDING:
            st.html(
                '<div class="ctai-check bad">⚠️ AI flagged this statement. '
                "Please review before approving.</div>"
            )

        saved_text = item.plain_text
        unsaved_edit = False

        with st.expander("Edit wording"):
            new_text = st.text_area(
                "Patient-friendly wording",
                value=item.plain_text,
                key=f"{key}_text",
                height=90,
                label_visibility="collapsed",
            )
            unsaved_edit = new_text.strip() != saved_text
            if unsaved_edit:
                st.caption("You have unsaved changes. Click **Save Changes** to keep them, or Cancel to discard.")
            save_col, cancel_col, _ = st.columns([1, 1, 3])
            cancel_col.button(
                "Cancel",
                key=f"{key}_cancel",
                disabled=not unsaved_edit,
                on_click=_restore_text,
                args=(f"{key}_text", saved_text),
            )
            if save_col.button("Save Changes", key=f"{key}_save"):
                if not new_text.strip():
                    st.session_state[f"{key}_msg"] = ("error", "The statement cannot be empty.")
                elif not unsaved_edit:
                    st.session_state[f"{key}_msg"] = ("info", "No changes to save.")
                else:
                    guide = get_guide()
                    edit_item(item, new_text)
                    if guide is not None:
                        _clear_override_confirmation(guide, item)
                    pages = (st.session_state.get("pipeline_result") or {}).get(
                        "source_pages"
                    ) or {}
                    remaining = reverify_item(item, pages.get(str(item.source_page)))
                    if persist_review("edited", f"{_SECTION_LABEL.get(item.section, item.section)}: {item.plain_text}"):
                        if remaining:
                            st.session_state[f"{key}_msg"] = (
                                "warning",
                                "Edit saved. Please review the AI flag before approving.",
                            )
                        else:
                            st.session_state[f"{key}_msg"] = (
                                "success",
                                "Edit saved and checked against the protocol. You can approve it.",
                            )
                st.rerun()

        if item.source_status != "verified" and not item.rejected:
            with st.expander("Inspect source", expanded=False):
                _render_source_panel(item, key)

        block_reason = approval_block_reason(item)
        col_yes, col_no, col_undo, _ = st.columns([1.3, 1.3, 1.3, 1.1])

        if decided:
            if col_yes.button("Withdraw approval", key=f"{key}_unapprove"):
                guide = get_guide()
                unapprove_item(item)
                if guide is not None:
                    _clear_override_confirmation(guide, item)
                if persist_review("approval_withdrawn", item.plain_text):
                    st.session_state[f"{key}_msg"] = ("info", "Approval withdrawn.")
                st.rerun()
        else:
            if col_yes.button(
                "Approve",
                key=f"{key}_approve",
                type="primary",
                disabled=block_reason is not None or unsaved_edit,
            ):
                try:
                    # Approve on a flagged statement is the explicit override.
                    approve_item(item)
                except ValueError as exc:
                    st.session_state[f"{key}_msg"] = ("error", str(exc))
                else:
                    guide = get_guide()
                    if item.override and guide is not None:
                        _mark_override_confirmed(guide, item)
                        saved = persist_review(
                            "approved_with_override", _override_audit_detail(item)
                        )
                    else:
                        saved = persist_review("approved", item.plain_text)
                    if saved:
                        st.session_state[f"{key}_msg"] = (
                            "success",
                            "Approved. It will reach patients when you publish.",
                        )
                st.rerun()
            if block_reason:
                st.caption("Cannot approve: " + block_reason)
            elif unsaved_edit:
                st.caption("Save your edit before approving.")
        if state == REJECTED:
            if col_undo.button("↩ Reconsider", key=f"{key}_reconsider"):
                reconsider_item(item)
                if persist_review("reconsidered", item.plain_text):
                    st.session_state[f"{key}_msg"] = ("info", "Statement returned to Pending.")
                st.rerun()
        else:
            with col_no.container(key=f"{key}_reject_style"):
                if st.button("Reject", key=f"{key}_reject"):
                    guide = get_guide()
                    reject_item(item)
                    if guide is not None:
                        _clear_override_confirmation(guide, item)
                    if persist_review("rejected", item.plain_text):
                        st.session_state[f"{key}_msg"] = (
                            "success",
                            "Rejected. It will never be shown to patients.",
                        )
                    st.rerun()

        message = st.session_state.pop(f"{key}_msg", None)
        if message:
            getattr(st, message[0])(message[1])


def _recheck_guide(pipeline_result: Mapping[str, Any], guide: PatientGuide) -> int:
    """Rerun the deterministic source check and apply it to unverified statements."""
    pages = pipeline_result.get("source_pages")
    rows = pipeline_result.get("verification")
    if pages:
        rows = verify_saved_pages(pipeline_result["analysis"], pages)
        try:
            pipeline_result["verification"] = rows
        except TypeError:
            pass
    return recheck_sources(guide, rows)


def _render_source_panel(item: GuideItem, key: str) -> None:
    """Optional source inspection: cited page text for coordinator review."""
    pages = (st.session_state.get("pipeline_result") or {}).get("source_pages") or {}
    result = inspect_source(item, pages)
    st.html(f'<div class="ctai-label">Protocol {page_label(item) or "page ?"}</div>')
    if result["page_text"]:
        st.text_area(
            "Cited protocol page",
            value=result["page_text"],
            height=180,
            disabled=True,
            key=f"{key}_pagetext",
        )
    elif result["lines"]:
        for line in result["lines"]:
            st.html(f'<div class="ctai-quote">{theme.e(line)}</div>')
    else:
        st.caption("No protocol page text is available to inspect for this statement.")


def render_guide_review_tab(pipeline_result: Mapping[str, Any]) -> bool:
    """
    Sections 1 and 2 of the Review page: review each statement, then add one.

    Statements use the full page width under a compact status bar. Returns
    False while the statements are not available yet.
    """
    guide = _ensure_guide(pipeline_result)
    if guide is None:
        return False

    counts = state_counts(guide)
    total = counts[ALL]
    expected = max(guide.generation_total or total, total)
    generating = not guide.generation_complete
    pending = counts[PENDING]
    accepted = counts[APPROVED] + counts[EDITED]
    rejected = counts[REJECTED]
    needs_source_check = any(
        item.source_status == "unchecked" for item in guide.all_items()
    )

    pending_items = [
        i for i in guide.all_items()
        if review_state(i) == PENDING and approval_block_reason(i) is None
    ]
    flagged_pending = sum(
        1 for i in pending_items
        if verification_warnings(i) or i.needs_review
    )
    approvable = len(pending_items)
    decided = accepted + rejected
    active_filter = st.session_state.setdefault(FILTER_KEY, ALL)

    # Sticky review-status bar: clickable counts on the left, Approve all on the right.
    bar = st.container(key="review_bar")
    with bar:
        stats, approve_col = st.columns([6, 2], vertical_alignment="center")
        total_text = f"{total} / {expected}" if generating else str(total)
        filters = (
            (ALL, "Total", total_text),
            (APPROVED, "Approved", accepted),
            (PENDING, "Pending", pending),
            (REJECTED, "Rejected", rejected),
            (EDITED, "Edited", counts["edited_total"]),
        )
        with stats.container(key="guide_filters"):
            cols = st.columns(len(filters))
            for col, (key, label, value) in zip(cols, filters):
                with col:
                    selected = active_filter == key
                    if st.button(
                        f"{label} {value}",
                        key=f"guide_filter_{key}",
                        type="primary" if selected else "secondary",
                    ):
                        st.session_state[FILTER_KEY] = key
                        st.rerun()
        with approve_col:
            if approvable == 0:
                st.session_state.pop(APPROVE_ALL_CONFIRM_KEY, None)
                st.button(
                    "Approve all",
                    key="guide_approve_all_btn",
                    disabled=True,
                    use_container_width=True,
                )
            elif st.session_state.get(APPROVE_ALL_CONFIRM_KEY):
                st.caption(
                    f"Approve **{approvable}** pending"
                    + (f" (**{flagged_pending}** AI-flagged)" if flagged_pending else "")
                    + "?"
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    "Confirm",
                    key="guide_approve_all_confirm_btn",
                    type="primary",
                    use_container_width=True,
                ):
                    checked = approve_all_unflagged(guide)
                    overridden = approve_all_with_override(guide)
                    user = authz.current_user() or {}
                    actor = user.get("email") or user.get("user_id") or "coordinator"
                    persist_review(
                        "approved_all_checked",
                        f"{checked} statement(s) | actor={actor}",
                    )
                    if overridden:
                        persist_review(
                            "approved_all_with_override",
                            f"{overridden} statement(s) | actor={actor} | status=flagged_approved",
                        )
                    st.session_state.pop(APPROVE_ALL_CONFIRM_KEY, None)
                    st.session_state["guide_recheck_msg"] = (
                        f"Approved {checked + overridden} pending statement(s)"
                        + (f" ({overridden} with AI override)" if overridden else "")
                        + "."
                    )
                    st.rerun()
                if cancel_col.button(
                    "Cancel",
                    key="guide_approve_all_cancel_btn",
                    use_container_width=True,
                ):
                    st.session_state.pop(APPROVE_ALL_CONFIRM_KEY, None)
                    st.rerun()
            else:
                if st.button(
                    "Approve all",
                    key="guide_approve_all_btn",
                    use_container_width=True,
                ):
                    st.session_state[APPROVE_ALL_CONFIRM_KEY] = True
                    st.rerun()
        pct = max(0, min(100, int(round(100 * decided / expected)))) if expected else 0
        st.html(f'<div class="ctai-rbar-progress"><div style="width:{pct}%"></div></div>')

    st.caption(
        "Approve, edit or reject each statement. Rejected statements are never "
        "shown to patients."
    )
    if generating:
        st.caption("Statements are still being generated. Publishing waits until they finish.")
    message = st.session_state.pop("guide_recheck_msg", None)
    if message:
        st.info(message)

    visible = 0
    number = 0
    for section in guide.sections():
        section_items = [
            item for item in section.items if matches_filter(item, active_filter)
        ]
        if section.unavailable_message and active_filter == ALL:
            number += len(section.items)
            st.subheader(section.title)
            st.caption(section.unavailable_message)
            continue
        if not section_items:
            number += len(section.items)
            continue
        st.subheader(section.title)
        for item in section.items:
            number += 1
            if item not in section_items:
                continue
            visible += 1
            _render_review_item(
                item,
                key=f"guide_{item.item_id}",
                number=number,
                total=total,
            )
    if visible == 0:
        st.info("No statements match this filter.")

    render_add_statement(guide, pipeline_result, persist_review)

    with st.expander("More options", expanded=False):
        if needs_source_check:
            if pipeline_result.get("source_pages") or pipeline_result.get("verification"):
                if st.button("🔎 Recheck sources", key="guide_recheck_btn"):
                    resolved = _recheck_guide(pipeline_result, guide)
                    if persist_review("source_rechecked", f"{resolved} statement(s)"):
                        st.session_state["guide_recheck_msg"] = (
                            f"Source check ran: {resolved} statement(s) changed status."
                        )
                    st.rerun()
            else:
                st.caption(
                    "Re-run the protocol analysis (cached results are reused) to "
                    "provide the current source text for the AI source check."
                )
        st.download_button(
            "Download guide (Markdown)",
            data=guide_to_markdown(patient_visible_guide(guide)),
            file_name="patient_guide.md",
            mime="text/markdown",
            key="guide_download_btn",
        )
        if st.button("Regenerate guide", key="guide_regenerate_btn"):
            try:
                review_store.clear_generated_items(_store_id())
            except (sqlite3.Error, OSError):
                pass
            # Statements typed in by the coordinator are not AI output: keep them.
            st.session_state[MANUAL_KEEP_KEY] = [
                i for i in guide.all_items() if i.added_by_coordinator
            ]
            clear_guide()
            st.rerun()
        st.caption("Regenerating replaces the AI statements and their decisions; statements you added are kept.")
    return True


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
