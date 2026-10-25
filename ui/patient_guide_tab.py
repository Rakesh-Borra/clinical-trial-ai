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
import time
from typing import Any, Mapping, Optional

import streamlit as st

from agents.plain_language_agent import (
    REJECTED_REASON,
    active_overrides,
    approve_all_unflagged,
    approve_all_with_override,
    approval_block_reason,
    approve_item,
    item_cleared,
    overridable_items,
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
from models.plain_language_models import GuideItem, PatientGuide, page_label
from services import benchmark_log, guide_jobs, review_store
from services.guide_review import (
    ALL,
    APPROVED,
    EDITED,
    FILTERS,
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
from ui import theme

GUIDE_KEY = "patient_guide"
GUIDE_SOURCE_KEY = "patient_guide_source"
GUIDE_JOB_KEY = "guide_job_id"
GUIDE_ERROR_KEY = "guide_generation_error"
GUIDE_AUTOSTART_KEY = "guide_autostart_fingerprint"
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
    return "⏳ Waiting for your decision"


def get_guide() -> Optional[PatientGuide]:
    return st.session_state.get(GUIDE_KEY)


def is_fully_approved_snapshot(guide: Any) -> bool:
    """Validate that a patient-facing guide contains only complete approvals."""
    if (
        not isinstance(guide, PatientGuide)
        or guide.analysis_complete is not True
        or not guide.generation_complete
    ):
        return False
    items = guide.all_items()
    return (
        bool(items)
        and all(
            item.analysis_complete is True
            and item.approved
            and not item.rejected
            and item_cleared(item)
            for item in items
        )
        and sorted(i.item_id for i in active_overrides(guide))
        == sorted(guide.overrides_confirmed)
    )


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
    if state in (APPROVED, EDITED) and item.override:
        return theme.chip("Approved · coordinator override", "sun")
    if state == EDITED:
        return theme.chip("Edited · approved", "teal")
    if state == APPROVED:
        return theme.chip("Approved", "teal")
    if state == REJECTED:
        return theme.chip("Rejected", "coral")
    return theme.chip("Pending · waiting for your decision", "grey")


def _render_review_item(
    item: GuideItem,
    key: str,
    number: int = 0,
    total: int = 0,
) -> None:
    with st.container(border=True):
        page = f"Protocol {page_label(item)}" if item.source_page else "Page not found"
        count = f"Statement {number} of {total} · " if number else ""
        st.html(
            '<div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">'
            f'<span class="ctai-label" style="margin:0">{count}{theme.e(_SECTION_LABEL.get(item.section, item.section))} · {theme.e(page)}</span>'
            f'{_status_chip(item, _published_texts())}</div>'
        )
        st.html(f'<div class="ctai-label">What the protocol says</div><div class="ctai-quote">{theme.e(item.original_text)}</div>')
        st.html(f'<div class="ctai-label">What patients would read</div><div class="ctai-plain">{theme.e(item.plain_text)}</div>')

        state = review_state(item)
        warnings = verification_warnings(item)
        _render_verification_details(item, key, warnings, expanded=bool(warnings) and state == PENDING)

        saved_text = item.plain_text
        unsaved_edit = False

        with st.expander("Edit the wording"):
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
                    edit_item(item, new_text)
                    pages = (st.session_state.get("pipeline_result") or {}).get(
                        "source_pages"
                    ) or {}
                    remaining = reverify_item(item, pages.get(str(item.source_page)))
                    if persist_review("edited", f"{_SECTION_LABEL.get(item.section, item.section)}: {item.plain_text}"):
                        if remaining:
                            st.session_state[f"{key}_msg"] = (
                                "warning",
                                "Edit saved. The AI check still has warnings: "
                                + " ".join(remaining)
                                + " You can approve it anyway as a recorded override.",
                            )
                        else:
                            st.session_state[f"{key}_msg"] = (
                                "success",
                                "Edit saved and checked against the protocol. You can approve it.",
                            )
                st.rerun()

        if item.verification != "verified" and not item.rejected:
            if st.button("🔁 Reverify against protocol", key=f"{key}_reverify"):
                pages = (st.session_state.get("pipeline_result") or {}).get(
                    "source_pages"
                ) or {}
                problems = reverify_item(item, pages.get(str(item.source_page)))
                if persist_review("reverified" if not problems else "reverify_failed", item.plain_text):
                    if problems:
                        st.session_state[f"{key}_msg"] = (
                            "warning",
                            "Reverification found: " + " ".join(problems),
                        )
                    else:
                        st.session_state[f"{key}_msg"] = (
                            "success",
                            "Reverified against the protocol. You can now approve it.",
                        )
                st.rerun()

        block_reason = approval_block_reason(item)
        decided = state in (APPROVED, EDITED)
        override_reason = ""
        override_ready = True
        if not decided and warnings and not block_reason:
            st.caption(
                "Approving despite the AI warning is your decision and is "
                "recorded in the audit history as a human override."
            )
            acknowledged = st.checkbox(
                "I checked the protocol source and accept the AI verification warning",
                key=f"{key}_ack",
            )
            override_reason = st.text_input(
                "Reason for approving despite the warning",
                key=f"{key}_reason",
            )
            override_ready = acknowledged and bool(override_reason.strip())

        st.html('<div class="ctai-label" style="margin-top:.6rem">Is this statement accurate?</div>')
        col_yes, col_no, col_undo, _ = st.columns([1.3, 1.3, 1.3, 1.1])

        if decided:
            if col_yes.button("Withdraw approval", key=f"{key}_unapprove"):
                unapprove_item(item)
                if persist_review("approval_withdrawn", item.plain_text):
                    st.session_state[f"{key}_msg"] = ("info", "Approval withdrawn.")
                st.rerun()
        else:
            if col_yes.button(
                "✅ Yes, approve",
                key=f"{key}_approve",
                type="primary",
                disabled=block_reason is not None or unsaved_edit or not override_ready,
            ):
                try:
                    approve_item(item, override_reason or None)
                except ValueError as exc:
                    st.session_state[f"{key}_msg"] = ("error", str(exc))
                else:
                    if item.override:
                        detail = (
                            f"{item.plain_text} | override reason: {item.override_reason}"
                            f" | warnings: {' / '.join(item.override_warnings)}"
                        )
                        saved = persist_review("approved_with_override", detail)
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
            elif not override_ready:
                st.caption("Tick the acknowledgement and give a reason to approve despite the warning.")
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

        message = st.session_state.pop(f"{key}_msg", None)
        if message:
            getattr(st, message[0])(message[1])

        if item.terms_explained:
            st.caption(
                "Words explained for patients: "
                + ", ".join(display_term(t.term) for t in item.terms_explained)
            )


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


def _render_verification_details(
    item: GuideItem, key: str, warnings: list[str], expanded: bool
) -> None:
    """AI verification results: information for the coordinator, not a decision."""
    if warnings:
        st.html(
            '<div class="ctai-check bad">⚠️ AI verification found '
            f"{len(warnings)} warning(s). This is information; the decision is yours.</div>"
        )
    else:
        st.html('<div class="ctai-check ok">✔ AI verification passed: numbers kept, no medical advice or eligibility wording, source confirmed</div>')
    with st.expander("AI verification details", expanded=expanded):
        label = {
            "verified": "Verified against the cited protocol page",
            "unverified": "Not verified against the cited protocol page",
            "unchecked": "Not yet checked against the protocol source",
        }[item.source_status]
        st.caption(f"Source check: {label}")
        for warning in warnings:
            st.html(f'<div class="ctai-check bad">⚠️ {theme.e(warning)}</div>')
        for note in item.review_reasons:
            if note not in warnings and note != REJECTED_REASON:
                st.caption(f"AI note: {note}")
        if item.override:
            st.html(
                '<div class="ctai-check bad">Human override: '
                f"{theme.e(item.override_reason or '')}</div>"
            )
        if item.source_status != "verified":
            _render_source_panel(item, key)


def _render_source_panel(item: GuideItem, key: str) -> None:
    """Show the cited protocol text and why the statement failed its check."""
    pages = (st.session_state.get("pipeline_result") or {}).get("source_pages") or {}
    result = inspect_source(item, pages)
    st.html(f'<div class="ctai-label">Protocol {page_label(item) or "page ?"}</div>')
    st.caption(result["explanation"])
    if item.source_issue:
        st.caption(f"Check result: {item.source_issue}")
    if result["lines"]:
        st.html('<div class="ctai-label">Closest lines on that page</div>')
        for line in result["lines"]:
            st.html(f'<div class="ctai-quote">{theme.e(line)}</div>')
    if result["page_text"]:
        st.text_area(
            "Full text of the cited page",
            value=result["page_text"],
            height=180,
            disabled=True,
            key=f"{key}_pagetext",
        )
    st.caption(
        "To clear the warning, edit the wording to match the protocol and "
        "save; or approve it as a recorded override. Statements are never "
        "marked verified automatically."
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


def _render_override_confirmation(guide: PatientGuide) -> None:
    """Publication needs the set of human overrides confirmed explicitly."""
    overrides = active_overrides(guide)
    if not overrides:
        return
    ids = sorted(i.item_id for i in overrides)
    st.html(
        f'<div class="ctai-label">Human overrides to confirm ({len(overrides)})</div>'
    )
    for item in overrides:
        st.caption(
            f"“{item.plain_text}” (page {item.source_page or '?'}) — reason: "
            f"{item.override_reason}. Warning: {' / '.join(item.override_warnings)}"
        )
    if ids == sorted(guide.overrides_confirmed):
        st.success("These overrides are confirmed for publication.")
        return
    confirmed = st.checkbox(
        "I confirm these statements were approved over AI verification warnings "
        "after I checked the protocol source",
        key="guide_overrides_ack",
    )
    if st.button("Confirm overrides", key="guide_overrides_confirm_btn", disabled=not confirmed):
        guide.overrides_confirmed = ids
        if persist_review("overrides_confirmed", f"{len(ids)} override(s)"):
            st.session_state["guide_recheck_msg"] = "Overrides confirmed."
        st.rerun()


def render_guide_review_tab(pipeline_result: Mapping[str, Any]) -> Optional[Any]:
    """
    Coordinator screen: statements are generated on entry, then reviewed.

    Statements use the full page width under a sticky review-status bar. The
    column reserved for the Publish button is returned so the caller can fill
    it with the existing publication action.
    """
    guide = _ensure_guide(pipeline_result)
    if guide is None:
        return None

    counts = state_counts(guide)
    total = counts[ALL]
    expected = max(guide.generation_total or total, total)
    generating = not guide.generation_complete
    pending = counts[PENDING]
    accepted = counts[APPROVED] + counts[EDITED]
    rejected = counts[REJECTED]
    unverified = sum(
        1 for item in guide.all_items()
        if item.source_status != "verified" and not item.rejected
    )
    needs_source_check = any(
        item.source_status == "unchecked" for item in guide.all_items()
    )

    eligible = sum(
        1 for item in guide.all_items()
        if not item.approved and not item.rejected
        and approval_block_reason(item) is None
        and not verification_warnings(item)
        and not item.needs_review
    )
    decided = accepted + rejected

    # Sticky review-status bar: live counts on the left, actions on the right.
    bar = st.container(key="review_bar")
    with bar:
        stats, approve_col, publish_col = st.columns(
            [6, 1.7, 2.3], vertical_alignment="center"
        )
        total_text = f"{total} / {expected}" if generating else str(total)
        cells = [
            ("Total", total_text, ""),
            ("Approved", accepted, "ok"),
            ("Pending", pending, "warn" if pending else ""),
            ("Rejected", rejected, "bad" if rejected else ""),
            ("Edited", counts["edited_total"], ""),
        ]
        stats.html(
            '<div class="ctai-rbar-stats">'
            + "".join(
                f'<span class="ctai-rstat {cls}">{label}<b>{value}</b></span>'
                for label, value, cls in cells
            )
            + "</div>"
        )
        with approve_col.popover("Approve all", use_container_width=True):
            if st.button(
                f"Approve all statements that passed verification ({eligible})",
                key="guide_approve_all_btn",
                disabled=eligible == 0,
                use_container_width=True,
            ):
                approved_now = approve_all_unflagged(guide)
                persist_review("approved_all_checked", f"{approved_now} statement(s)")
                st.session_state["guide_recheck_msg"] = (
                    f"Approved {approved_now} statement(s) that passed verification. "
                    "Statements with warnings were left for your decision."
                )
                st.rerun()
            warned = overridable_items(guide)
            if warned:
                st.caption(
                    f"{len(warned)} statement(s) have verification warnings. "
                    "Approving them is your decision and is recorded as a "
                    "human override for each one."
                )
                bulk_ack = st.checkbox(
                    "I reviewed these statements against the protocol and accept their warnings",
                    key="guide_bulk_override_ack",
                )
                bulk_reason = st.text_input(
                    "Reason for approving them despite the warnings",
                    key="guide_bulk_override_reason",
                )
                if st.button(
                    f"Approve {len(warned)} with override",
                    key="guide_bulk_override_btn",
                    disabled=not (bulk_ack and bulk_reason.strip()),
                    use_container_width=True,
                ):
                    count = approve_all_with_override(guide, bulk_reason)
                    persist_review(
                        "approved_all_with_override",
                        f"{count} statement(s) | reason: {bulk_reason.strip()}",
                    )
                    st.session_state["guide_recheck_msg"] = (
                        f"Approved {count} statement(s) as human overrides."
                    )
                    st.rerun()
        pct = max(0, min(100, int(round(100 * decided / expected)))) if expected else 0
        st.html(f'<div class="ctai-rbar-progress"><div style="width:{pct}%"></div></div>')

    st.caption(
        "The AI rewrites each protocol statement in plain language and runs "
        "automatic checks, shown as verification details. You decide on every "
        "statement: approve, edit and approve, or reject. Approving over a "
        "verification warning needs your reason and is recorded as a human "
        "override. Rejected statements stay here for audit but are never shown "
        "to patients."
    )
    if guide.analysis_complete is not True:
        st.warning(
            "The analysis did not finish cleanly, so publication is blocked. "
            "You can still review every statement."
        )
    if unverified:
        st.caption(
            f"{unverified} statement(s) have AI source-verification warnings "
            "(see each statement's AI verification details)."
        )
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
    message = st.session_state.pop("guide_recheck_msg", None)
    if message:
        st.info(message)
    if generating:
        st.caption(
            "Generation is still running or incomplete: publication stays "
            "blocked until every statement exists and has a decision."
        )
    if pending:
        st.info(f"{pending} statement(s) still need your decision.")
    if not pending and not generating:
        if accepted:
            st.success(
                f"Every statement has a decision: {accepted} to publish, "
                f"{rejected} rejected. Rejected statements are never shown to patients."
            )
        else:
            st.warning("Every statement is rejected. Approve at least one to publish.")
    _render_override_confirmation(guide)

    b2, b3, _ = st.columns([1, 1, 2])
    b2.download_button(
        "Download guide (Markdown)",
        data=guide_to_markdown(patient_visible_guide(guide)),
        file_name="patient_guide.md",
        mime="text/markdown",
        key="guide_download_btn",
    )
    if b3.button("Regenerate guide", key="guide_regenerate_btn"):
        try:
            review_store.clear_generated_items(_store_id())
        except (sqlite3.Error, OSError):
            pass
        clear_guide()
        st.rerun()

    selected = _render_filters(counts)
    if counts[selected] == 0 and selected != ALL:
        st.caption("No statements match this filter.")

    number = 0
    for section in guide.sections():
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
        for item in section.items:
            number += 1
            if matches_filter(item, selected):
                _render_review_item(
                    item,
                    key=f"guide_{item.item_id}",
                    number=number,
                    total=total,
                )
    return publish_col


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
