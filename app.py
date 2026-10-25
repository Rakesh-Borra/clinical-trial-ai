"""
Clinical Trial AI — Streamlit MVP.

Wires Phase 3 pipeline + Phase 4/5 assistants. Analysis runs once per
upload and is cached in session state. Does not modify backend modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import streamlit as st

from models.plain_language_models import PatientGuide, as_patient_guide
from models.protocol_models import ProtocolAnalysis
from agents.guide_patient_agent import APPROVED_STATUS
from models.protocol_models import ProtocolAnalysis
from services import analysis_jobs, authz, guide_jobs, review_store, version_compare, version_store
from services.model_provider import provider_display_name
from services.protocol_pipeline import analyze_protocol_pdf
from services import benchmark_log
from ui import auth_pages, coordinator_home, theme
from ui.landing import (
    COORD_VIEW_KEY,
    COORDINATOR,
    NAME_KEY,
    PATIENT,
    ROLE_KEY,
    VIEW_HOME,
    VIEW_LIST,
    VIEW_VERSIONS,
    VIEW_WORKFLOW,
    coordinator_page,
    navigate_to_page,
    patient_page,
    record_page_visit,
    render_landing,
    render_top_bar,
)
from services.coverage import build_coverage, missing_categories
from services.guide_review import build_published_snapshot, decision_block_reason
from services.summary_review import is_summary_approved, summary_publish_block_reason
from ui.summary_panel import (
    render_summary_generation,
    render_summary_panel,
)
from agents.plain_language_agent import guide_to_markdown
from ui.patient_guide_tab import (
    GUIDE_JOB_KEY,
    GUIDE_KEY,
    GUIDE_SOURCE_KEY,
    analysis_fingerprint,
    clear_guide,
    get_guide,
    final_validation_reason,
    is_fully_approved_snapshot,
    persist_review,
    purge_item_widget_state,
    render_guide_review_tab,
)
from ui.patient_view import render_patient_workspace
from ui.version_panel import render_version_panel

RESEARCH_DISCLAIMER = (
    "Research prototype for educational demonstration only. This tool does "
    "not provide medical advice, diagnose conditions, prescribe treatment, "
    "or determine whether any person is eligible for a clinical trial. "
    "Always discuss trial participation with the study team and a qualified "
    "clinician. AI-extracted content requires human review and does not "
    "replace the full protocol or informed consent."
)

# Purely informational pipeline notes — hide from user-facing UI.
_INFORMATIONAL_WARNING_MARKERS = (
    "rescue skipped",
    "criteria appear complete",
)

_SAFETY_MONITORING_MARKERS = (
    "monitored for adverse",
    "monitor for adverse",
    "adverse event monitoring",
    "monitoring for adverse",
    "will be monitored",
)

_SAFETY_RISK_MARKERS = (
    "risk",
    "hypoglycemia",
    "hyperglycemia",
    "side effect",
    "adverse reaction",
    "toxicity",
    "hazard",
    "serious adverse",
    "requires clinical review",
)

_ELIGIBILITY_INCLUSION_HEADER = "Requirements that may allow someone to join"
_ELIGIBILITY_EXCLUSION_HEADER = "Reasons someone may not be able to join"
_UNSPECIFIED_RISKS_NOTE = (
    "Specific study risk information was not specified in the extracted "
    "summary. That does not necessarily mean the full protocol lacks risk "
    "details—please ask the study team for complete safety information."
)

SESSION_KEYS = (
    "pipeline_result",
    "analysis_error",
    "temp_pdf_path",
    "uploaded_name",
    "uploaded_fingerprint",
    "coordinator_history",
    "patient_history",
)


def init_session_state() -> None:
    """Ensure required session keys exist."""
    defaults: dict[str, Any] = {
        "pipeline_result": None,
        "analysis_error": None,
        "temp_pdf_path": None,
        "uploaded_name": None,
        "uploaded_fingerprint": None,
        "coordinator_history": [],
        "patient_history": [],
        "coordinator_protocol_id": None,
        "coordinator_protocols": {},
        "published_protocols": {},
        "analysis_job_id": None,
        "patient_active_protocol_id": None,
        "patient_summary_open": False,
        "coord_max_step": 0,
        ROLE_KEY: None,
        NAME_KEY: "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_analysis_state(*, keep_upload: bool = False) -> None:
    """Drop cached analysis and Q&A history; optionally keep the upload path."""
    analysis_jobs.jobs.discard(st.session_state.get("analysis_job_id"))
    st.session_state["analysis_job_id"] = None
    cleanup_temp_pdf(st.session_state.get("temp_pdf_path"))
    st.session_state.pipeline_result = None
    st.session_state.analysis_error = None
    st.session_state.coordinator_history = []
    st.session_state.patient_history = []
    st.session_state.coordinator_protocol_id = None
    st.session_state.coord_max_step = 0
    clear_guide()
    if not keep_upload:
        st.session_state.temp_pdf_path = None
        st.session_state.uploaded_name = None
        st.session_state.uploaded_fingerprint = None


def cleanup_temp_pdf(path: Optional[str]) -> None:
    """Best-effort removal of a temporary uploaded PDF."""
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def persist_uploaded_pdf(uploaded_file: Any) -> str:
    """
    Write an uploaded Streamlit file to a temporary PDF path.

    Caller owns cleanup via cleanup_temp_pdf / clear_analysis_state.
    """
    suffix = Path(getattr(uploaded_file, "name", "protocol.pdf")).suffix or ".pdf"
    fd, temp_path = tempfile.mkstemp(prefix="ctai_protocol_", suffix=suffix)
    os.close(fd)
    with open(temp_path, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return temp_path


def upload_fingerprint(uploaded_file: Any) -> str:
    """Stable-enough key for detecting a new upload in the same session."""
    name = getattr(uploaded_file, "name", "unknown")
    size = getattr(uploaded_file, "size", None)
    fingerprint = f"{name}:{size}"
    get_buffer = getattr(uploaded_file, "getbuffer", None)
    if get_buffer is not None:
        digest = hashlib.sha256(get_buffer()).hexdigest()[:16]
        fingerprint = f"{fingerprint}:{digest}"
    return fingerprint


def run_protocol_analysis(pdf_path: str) -> dict[str, Any]:
    """
    Call the Phase 3 pipeline. Separated for mocking in tests.

    Raises ValueError for empty/invalid extractable content and may raise
    connection/runtime errors if the configured model provider is unavailable.
    """
    return analyze_protocol_pdf(pdf_path)


def active_analysis_job() -> Optional[analysis_jobs.JobSnapshot]:
    """The queued or running analysis job for this session, if any."""
    job = analysis_jobs.jobs.get(st.session_state.get("analysis_job_id"))
    return job if job is not None and job.active else None


def start_analysis() -> None:
    """Start analysis in the background for the current upload (idempotent)."""
    authz.require_coordinator()
    if st.session_state.get("pipeline_result") is not None or active_analysis_job():
        return
    st.session_state["analysis_error"] = None
    benchmark_log.name_file(
        st.session_state.temp_pdf_path, st.session_state.get("uploaded_name") or ""
    )
    st.session_state["analysis_job_id"] = analysis_jobs.jobs.submit(
        run_protocol_analysis,
        st.session_state.temp_pdf_path,
        st.session_state.coordinator_protocol_id,
        st.session_state.uploaded_fingerprint,
    )


def collect_analysis_job() -> bool:
    """
    Move a finished job's outcome into session state (main thread only).

    Results are attached only when the job still belongs to the current
    protocol and PDF; anything else is discarded. Returns True when a
    completed result was just stored.
    """
    job_id = st.session_state.get("analysis_job_id")
    job = analysis_jobs.jobs.get(job_id)
    if job is None:
        st.session_state["analysis_job_id"] = None
        return False
    if job.active:
        return False

    st.session_state["analysis_job_id"] = None
    analysis_jobs.jobs.release(job_id)
    if (
        job.status == analysis_jobs.DISCARDED
        or job.protocol_id != st.session_state.get("coordinator_protocol_id")
        or job.fingerprint != st.session_state.get("uploaded_fingerprint")
    ):
        return False
    if job.status == analysis_jobs.FAILED:
        st.session_state["pipeline_result"] = None
        st.session_state["analysis_error"] = (
            f"{format_analysis_error(job.error)}\n\n"
            f"Analysis stopped at {int(job.progress.percent)}% "
            f"during: {job.progress.stage}. Your uploaded protocol is kept; "
            "you can try again."
        )
        return False
    st.session_state["pipeline_result"] = job.result
    st.session_state["analysis_error"] = None
    return True


def _analysis_status_body() -> None:
    job = active_analysis_job()
    if job is None:
        # The coordinator watched it finish, so the app may advance to Summary.
        st.session_state["_analysis_watched"] = True
        st.rerun()
    progress = job.progress
    if job.status == analysis_jobs.QUEUED:
        st.progress(0, text="Waiting for another analysis to finish… (indeterminate)")
    else:
        percent = int(progress.percent)
        if progress.determinate and progress.total:
            detail = f"{progress.stage} ({progress.completed} of {progress.total})"
        else:
            detail = f"{progress.stage} (in progress)"
        st.progress(percent / 100, text=f"{percent}% complete — {detail}")
    st.info(
        f"Analyzing with {provider_display_name()}. Analysis keeps running if you "
        "leave this page. This can take several minutes; results are saved "
        "automatically when it finishes."
    )


@st.fragment(run_every=2)
def _poll_analysis_status() -> None:
    """Refresh the status and rerun the app once the job has finished."""
    _analysis_status_body()


def render_analysis_status() -> bool:
    """Show running-job status; returns True while analysis is in progress."""
    if active_analysis_job() is None:
        return False
    _poll_analysis_status()
    return True


def format_analysis_error(exc: BaseException) -> str:
    """Map backend failures to user-facing guidance."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.casefold()

    if isinstance(exc, ValueError) or "no extractable text" in lowered:
        return (
            "Could not analyze this PDF. It may be empty, unreadable, or "
            "have no extractable text after OCR fallback. "
            f"Details: {message}"
        )

    ollama_markers = (
        "connection refused",
        "failed to connect",
        "ollama",
        "httpconnectionpool",
        "max retries exceeded",
        "connection error",
    )
    if any(marker in lowered for marker in ollama_markers):
        if provider_display_name() == "Ollama (local)":
            return (
                "Protocol analysis failed because the local Ollama service was "
                "unreachable or the model could not be called. Confirm Ollama is "
                "running and that qwen3:4b is available, then try Analyze again. "
                f"Details: {message}"
            )

    if "openrouter" in lowered or provider_display_name() == "OpenRouter (cloud)":
        return (
            "Protocol analysis failed using OpenRouter. Check the provider "
            "configuration and service availability, then try Analyze again. "
            f"Details: {message}"
        )

    return f"Protocol analysis failed: {message}"


def analysis_completeness_message(
    result: Mapping[str, Any],
) -> tuple[bool, str]:
    """Describe completeness safely, treating absent or unknown state as partial."""
    analysis = result.get("analysis")
    is_complete = result.get(
        "analysis_complete",
        getattr(analysis, "analysis_complete", None),
    )
    if is_complete is True:
        return True, "Analysis complete. Results are cached for this upload."

    reasons = list(result.get("incomplete_reasons") or [])
    failed_chunks = result.get("incomplete_chunks") or []
    details = []
    if failed_chunks:
        details.append(f"{len(failed_chunks)} chunk(s) did not complete.")
    details.extend(str(reason) for reason in reasons if reason)
    explanation = " ".join(details) or (
        "We could not verify completeness: the pipeline did not confirm "
        "that all required analysis completed."
    )
    return (
        False,
        "Analysis is incomplete. Treat these results as partial; patient "
        "publication and approval remain blocked. " + explanation,
    )


def render_analysis_completeness(result: Mapping[str, Any]) -> bool:
    """Render durable complete/incomplete status and return completeness."""
    complete, message = analysis_completeness_message(result)
    if complete:
        st.success(message)
    else:
        st.warning(message)
    return complete


def verification_lookup(
    verification: list[dict],
) -> dict[tuple[str, str], dict]:
    """Index verification rows by (category, statement)."""
    index: dict[tuple[str, str], dict] = {}
    for row in verification:
        category = row.get("category")
        statement = row.get("statement")
        if category is None or statement is None:
            continue
        index[(str(category), str(statement))] = row
    return index


def _status_badge(status: Optional[str]) -> str:
    """Human-readable verification label for display only (internal codes unchanged)."""
    if not status:
        return "Unknown"
    labels = {
        "exact_match": "Verified",
        "not_verified": "Review",
        "timing_not_found": "Timing not found",
        "missing_page": "Missing page",
        "invalid_page": "Invalid page",
        "illustrative_only": "Example text only",
    }
    return labels.get(str(status), str(status).replace("_", " ").capitalize())


def _group_attention_warnings(warnings: list[str]) -> dict[str, list[str]]:
    """Bucket Summary 'Needs review' lines for a compact expander."""
    groups = {
        "Conflicting information": [],
        "Unsupported statements": [],
        "Duplicates": [],
        "Other warnings": [],
    }
    for warning in warnings:
        lower = warning.casefold()
        if "conflict" in lower:
            groups["Conflicting information"].append(warning)
        elif any(tok in lower for tok in ("not verified", "unsupported", "not found", "exact_match")):
            groups["Unsupported statements"].append(warning)
        elif "duplicate" in lower:
            groups["Duplicates"].append(warning)
        else:
            groups["Other warnings"].append(warning)
    return {name: items for name, items in groups.items() if items}


def is_informational_warning(warning: str) -> bool:
    """True for processing notes that should not clutter the UI."""
    normalized = warning.casefold()
    return any(marker in normalized for marker in _INFORMATIONAL_WARNING_MARKERS)


def filter_user_facing_warnings(warnings: Optional[Sequence[str]]) -> list[str]:
    """Keep genuine accuracy/extraction warnings; drop informational notes."""
    if not warnings:
        return []
    return [item for item in warnings if not is_informational_warning(item)]


def classify_safety_statement(statement: str) -> str:
    """
    Classify a safety statement as monitoring, risk, or other.

    Used only for display grouping — does not invent clinical content.
    """
    normalized = statement.casefold()
    monitoring_hit = any(marker in normalized for marker in _SAFETY_MONITORING_MARKERS)
    risk_hit = any(marker in normalized for marker in _SAFETY_RISK_MARKERS)

    if monitoring_hit and not risk_hit:
        return "monitoring"
    if risk_hit:
        return "risk"
    if "adverse event" in normalized or "adverse events" in normalized:
        if "monitor" in normalized:
            return "monitoring"
        return "other"
    return "other"


def _bullet_statement(statement: str, *, patient_style: bool) -> str:
    text = statement.strip()
    if patient_style:
        return f'- "{text}"'
    return f"- {text}"


def _extract_preface(answer_text: str) -> str:
    """Return prose lines before the first bullet or known section header."""
    lines: list[str] = []
    for raw in (answer_text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith("- "):
            break
        if stripped.rstrip(":").startswith(_ELIGIBILITY_INCLUSION_HEADER):
            break
        if stripped.rstrip(":").startswith(_ELIGIBILITY_EXCLUSION_HEADER):
            break
        if stripped.lower().startswith("verified inclusion"):
            break
        if stripped.lower().startswith("verified exclusion"):
            break
        lines.append(stripped)
    return "\n\n".join(lines).strip()


def _citations_by_category(citations: Sequence[Any], category: str) -> list[Any]:
    return [c for c in citations if getattr(c, "category", None) == category]


def format_display_answer_text(
    answer: Any,
    *,
    audience: Optional[str] = None,
) -> str:
    """
    Build a readable answer body for the UI.

    Preserves exact clinical statements from citations. Moves page numbers
    and verification statuses out of the main body (shown in expanders).
    """
    intent = getattr(answer, "intent", "")
    citations = list(getattr(answer, "citations", []) or [])
    answer_text = getattr(answer, "answer_text", "") or ""
    patient_style = (audience or getattr(answer, "audience", "")) == "patient"

    # Answers built from the approved Patient Guide are already written for
    # patients; show them as they are.
    if citations and all(
        getattr(c, "verification_status", "") == APPROVED_STATUS for c in citations
    ):
        return re.sub(r"\n{3,}", "\n\n", answer_text.strip())

    if intent in {"medical_advice_refused", "unsupported"} or not citations:
        # Keep agent prose for refusals/empty cases; just normalize spacing.
        return re.sub(r"\n{3,}", "\n\n", answer_text.strip())

    if intent == "eligibility_decision_refused":
        parts = [
            _extract_preface(answer_text)
            or (
                "I cannot tell you whether you personally are eligible for "
                "this study. Eligibility depends on your individual situation "
                "and must be decided with the study team."
                if patient_style
                else "I cannot determine whether a specific patient is "
                "eligible or provide a final enrollment decision."
            )
        ]
        inclusions = _citations_by_category(citations, "inclusion_criteria")
        exclusions = _citations_by_category(citations, "exclusion_criteria")

        if inclusions:
            header = (
                _ELIGIBILITY_INCLUSION_HEADER
                if patient_style
                else "Verified inclusion criteria"
            )
            parts.append(f"\n\n**{header}:**\n\n")
            parts.append(
                "\n".join(
                    _bullet_statement(c.statement, patient_style=patient_style)
                    for c in inclusions
                )
            )
        if exclusions:
            header = (
                _ELIGIBILITY_EXCLUSION_HEADER
                if patient_style
                else "Verified exclusion criteria"
            )
            parts.append(f"\n\n**{header}:**\n\n")
            parts.append(
                "\n".join(
                    _bullet_statement(c.statement, patient_style=patient_style)
                    for c in exclusions
                )
            )
        if not inclusions and not exclusions:
            parts.append(
                "\n\nNo verified inclusion or exclusion criteria were "
                "available in the extracted study summary."
            )
        return "".join(parts).strip()

    if intent == "safety":
        monitoring: list[Any] = []
        risks: list[Any] = []
        other: list[Any] = []
        for citation in citations:
            kind = classify_safety_statement(citation.statement)
            if kind == "monitoring":
                monitoring.append(citation)
            elif kind == "risk":
                risks.append(citation)
            else:
                other.append(citation)

        parts: list[str] = [
            "Safety-related information from the verified study summary "
            "(exact wording preserved):"
        ]
        if monitoring:
            parts.append("\n\n**Adverse-event monitoring:**\n\n")
            parts.append(
                "\n".join(
                    _bullet_statement(c.statement, patient_style=patient_style)
                    for c in monitoring
                )
            )
        if risks:
            parts.append("\n\n**Study risk information:**\n\n")
            parts.append(
                "\n".join(
                    _bullet_statement(c.statement, patient_style=patient_style)
                    for c in risks
                )
            )
        if other:
            parts.append("\n\n**Other safety-related statements:**\n\n")
            parts.append(
                "\n".join(
                    _bullet_statement(c.statement, patient_style=patient_style)
                    for c in other
                )
            )
        if not risks:
            parts.append(f"\n\n{_UNSPECIFIED_RISKS_NOTE}")
        return "".join(parts).strip()

    preface = _extract_preface(answer_text) or (
        "From the verified study summary (exact wording preserved):"
    )
    lines = [preface, ""]
    lines.extend(
        _bullet_statement(c.statement, patient_style=patient_style)
        for c in citations
    )
    return "\n".join(lines).strip()


def render_disclaimer() -> None:
    st.info(RESEARCH_DISCLAIMER)


def render_protocol_summary(pipeline_result: Mapping[str, Any]) -> None:
    """Show extracted protocol facts with verification and warnings."""
    analysis: ProtocolAnalysis = pipeline_result["analysis"]
    verification = list(pipeline_result.get("verification") or [])
    extraction_warnings = filter_user_facing_warnings(
        list(pipeline_result.get("extraction_warnings") or [])
    )
    accuracy_warnings = filter_user_facing_warnings(
        list(pipeline_result.get("accuracy_warnings") or [])
    )
    vindex = verification_lookup(verification)

    st.subheader("Study metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Title:** {analysis.trial_title or 'Not available in summary'}")
    with col2:
        st.markdown(
            f"**Protocol ID:** {analysis.protocol_id or 'Not available in summary'}"
        )

    # Routine automatic fixes (page numbers corrected, title recovered) are
    # collapsed; items that need a coordinator's attention stay visible.
    routine_markers = ("corrected source_page", "recovered from labeled source")
    attention = [f"Extraction: {w}" for w in extraction_warnings] + [
        f"Accuracy: {w}"
        for w in accuracy_warnings
        if not any(marker in w for marker in routine_markers)
    ]
    routine = [
        w for w in accuracy_warnings if any(marker in w for marker in routine_markers)
    ]
    if attention:
        st.markdown(f"**Needs review — {len(attention)} items**")
        st.caption(
            "Check these against the protocol before approving patient content. "
            "Blocking analysis problems remain listed under analysis status above."
        )
        with st.expander("View details", expanded=False):
            for group_name, items in _group_attention_warnings(attention).items():
                st.markdown(f"**{group_name}** ({len(items)})")
                for warning in items:
                    st.markdown(f"- {warning}")
    if routine:
        with st.expander(f"{len(routine)} automatic correction(s) made during analysis"):
            for warning in routine:
                st.markdown(f"- {warning}")

    def _render_verified_list(
        title: str,
        category: str,
        items: list[Any],
        statement_fn,
        page_fn,
    ) -> None:
        st.subheader(title)
        if not items:
            st.caption("Not available in the extracted study summary.")
            return
        for item in items:
            statement = statement_fn(item)
            page = page_fn(item)
            row = vindex.get((category, statement), {})
            status = _status_badge(row.get("status"))
            page_label = page if page is not None else "unassigned"
            st.markdown(
                f"- {statement}  \n"
                f"  _Source page: {page_label} · Verification: {status}_"
            )

    _render_verified_list(
        "Inclusion criteria",
        "inclusion_criteria",
        analysis.inclusion_criteria,
        lambda c: c.description,
        lambda c: c.source_page,
    )
    _render_verified_list(
        "Exclusion criteria",
        "exclusion_criteria",
        analysis.exclusion_criteria,
        lambda c: c.description,
        lambda c: c.source_page,
    )

    st.subheader("Visits")
    if not analysis.visits:
        st.caption("Not available in the extracted study summary.")
    else:
        for visit in analysis.visits:
            timing = visit.timing or "timing not stated"
            row = vindex.get(("visits", visit.visit_name), {})
            status = _status_badge(row.get("status"))
            page = visit.source_page if visit.source_page is not None else "unassigned"
            procs = (
                f" · Procedures: {', '.join(visit.procedures)}"
                if visit.procedures
                else ""
            )
            st.markdown(
                f"- **{visit.visit_name}** — {timing}{procs}  \n"
                f"  _Source page: {page} · Verification: {status}_"
            )

    _render_verified_list(
        "Procedures",
        "procedures",
        analysis.procedures,
        lambda p: p.name,
        lambda p: p.source_page,
    )
    _render_verified_list(
        "Safety information",
        "safety_information",
        analysis.safety_information,
        lambda s: s.description,
        lambda s: s.source_page,
    )

    with st.expander("Full source verification table"):
        if not verification:
            st.caption("No verification rows available.")
        else:
            st.dataframe(verification, use_container_width=True)


STEPS = ["1 · Upload", "2 · Summary", "3 · Review"]


def current_protocol_name() -> str:
    protocol_id = st.session_state.get("coordinator_protocol_id")
    record = st.session_state.get("coordinator_protocols", {}).get(protocol_id, {})
    name = record.get("name", "")
    return name.strip() if isinstance(name, str) else ""


def _same_study_ids() -> set[str]:
    """Other versions of the study the current protocol belongs to (or will join)."""
    ids: set[str] = set()
    try:
        protocol_id = st.session_state.get("coordinator_protocol_id")
        if protocol_id:
            ids |= version_store.sibling_ids(protocol_id)
        study_id = st.session_state.get("coord_study_id")
        if study_id:
            ids |= version_store.members(study_id)
    except (sqlite3.Error, OSError):
        pass
    return ids


def publish_block_reason(
    guide: Any,
    pipeline_result: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Explain the first unmet publication guardrail."""
    if pipeline_result is None:
        if active_analysis_job() is not None:
            return "Analysis is still running. Publishing is blocked until it completes."
        return "Run protocol analysis before publishing."

    complete, message = analysis_completeness_message(pipeline_result)
    if not complete:
        return message
    guide = as_patient_guide(guide)
    if guide is None:
        return "Generate patient statements before publishing."
    if guide.analysis_complete is not True:
        return "The patient guide is incomplete or unverified; regenerate it from a complete analysis."
    if guide.generation_complete is not True:
        return (
            "Statement generation is incomplete or paused. Finish or resume "
            "generation before publishing."
        )
    try:
        from services import guide_jobs
        from ui.patient_guide_tab import GUIDE_JOB_KEY, analysis_fingerprint

        fingerprint = analysis_fingerprint(pipeline_result)
        job = guide_jobs.jobs.get(st.session_state.get(GUIDE_JOB_KEY))
        if job is None:
            job = guide_jobs.jobs.find_active(
                f"{st.session_state.get('coordinator_protocol_id') or ''}:{fingerprint}"
            )
        if job is not None and job.active:
            return "Statement generation is still running. Publishing is blocked until it finishes."
    except Exception:
        pass

    items = guide.all_items()
    if not items:
        return "No patient-facing statements are available to publish."

    # Rejected statements are excluded from publication, so only the ones
    # that would be published must have verified, complete analysis.
    incomplete = sum(
        item.analysis_complete is not True for item in items if not item.rejected
    )
    if incomplete:
        return (
            f"Cannot publish: {incomplete} statement(s) failed completeness "
            "verification."
        )

    reason = decision_block_reason(guide)
    if reason:
        return reason

    unsaved = [t for t in unsaved_work()[0] if "protocol name" not in t]
    if unsaved:
        return f"Save or cancel your edits first: {unsaved[0]}."

    protocol_id = st.session_state.get("coordinator_protocol_id")
    if not protocol_id:
        # Coverage overrides and the approved summary are stored per protocol.
        return "Cannot publish: the protocol has no active identifier. Upload it again."
    try:
        snapshot = build_published_snapshot(guide)
    except TypeError:
        return "The patient guide did not pass final approval validation: The patient guide could not be read."
    try:
        reason = summary_publish_block_reason(
            review_store.load_summary(protocol_id), snapshot
        ) or version_compare.comparison_block_reason(protocol_id, guide)
        if reason:
            return reason
        missing, fingerprint = acknowledgment_context(guide, pipeline_result, protocol_id)
        if missing and not review_store.acknowledgment_valid(protocol_id, fingerprint):
            return (
                "Missing categories: " + "; ".join(missing)
                + ". Tick the acknowledgment below to publish with incomplete information."
            )
    except sqlite3.Error as exc:
        return f"Cannot publish: review data could not be read ({exc})."

    failed = final_validation_reason(snapshot)
    if failed:
        return "The patient guide did not pass final approval validation: " + failed
    return None


def acknowledgment_context(
    guide: Any, pipeline_result: Mapping[str, Any], protocol_id: str
) -> tuple[list[str], str]:
    """
    The critical categories still missing, and a fingerprint of what the coordinator
    would be accepting: the version, the publishable statements, the summary
    text and the list of gaps. Any change to these invalidates an acknowledgment.
    """
    snapshot = build_published_snapshot(guide)
    missing = missing_categories(
        build_coverage(pipeline_result, snapshot),
        review_store.load_coverage_overrides(protocol_id),
        pipeline_result.get("analysis_complete") is True,
    )
    summary = review_store.load_summary(protocol_id) or {}
    version = version_store.get_version(protocol_id) or {}
    payload = json.dumps(
        {
            "protocol": protocol_id,
            "version": version.get("version_label"),
            "guide": _guide_content(snapshot),
            "summary": summary.get("draft_text"),
            "approved_summary": summary.get("approved_text"),
            "missing": missing,
        },
        sort_keys=True,
        default=str,
    )
    return missing, hashlib.sha256(payload.encode()).hexdigest()


def can_publish_guide(guide: Any, pipeline_result: Optional[Mapping[str, Any]]) -> bool:
    """Require verified complete analysis and coordinator approval for every item."""
    return publish_block_reason(guide, pipeline_result) is None


PUBLISHED_MESSAGE = "Published successfully! Coordinator task completed."


def published_record(protocol_id: Optional[str]) -> Optional[Mapping[str, Any]]:
    """Return the patient-visible snapshot for a protocol, or None.

    This is the single source of truth for "is it published": the snapshot must
    exist and pass the same approval validation the patient portal applies.
    """
    if not protocol_id:
        return None
    record = st.session_state.get("published_protocols", {}).get(protocol_id)
    if not isinstance(record, Mapping):
        return None
    name = record.get("name")
    if not (isinstance(name, str) and name.strip()):
        return None
    return record if is_fully_approved_snapshot(record.get("guide")) else None


def _guide_content(guide: Any) -> Optional[dict]:
    guide = as_patient_guide(guide)
    if guide is None:
        return None
    content = guide.model_dump()
    content.pop("trial_title", None)
    content.pop("protocol_id", None)
    return content


def _approved_summary_text() -> Optional[str]:
    protocol_id = st.session_state.get("coordinator_protocol_id")
    if not protocol_id:
        return None
    try:
        record = review_store.load_summary(protocol_id)
    except sqlite3.Error:
        return None
    return record["approved_text"] if is_summary_approved(record) else None


def _publishable(guide: Any) -> Any:
    """What publishing this guide would save (rejected statements removed)."""
    guide = as_patient_guide(guide)
    if guide is None:
        return None
    try:
        snapshot = build_published_snapshot(guide)
    except TypeError:
        return None
    snapshot.summary = _approved_summary_text()
    return snapshot


def published_matches_current(
    record: Mapping[str, Any], guide: Any, protocol_name: str
) -> bool:
    """True when the live guide and name are exactly what was published."""
    published = _guide_content(record.get("guide"))
    return (
        published is not None
        and published == _guide_content(_publishable(guide))
        and record.get("name") == protocol_name
    )


def publish_current_protocol() -> tuple[bool, str]:
    """Save an approved snapshot for the patient portal after all checks pass."""
    try:
        authz.require_coordinator()
    except authz.AuthorizationError as exc:
        return False, str(exc)
    protocol_id = st.session_state.get("coordinator_protocol_id")
    protocol_name = current_protocol_name()
    guide = get_guide()
    if not protocol_id:
        return False, "The uploaded protocol has no active identifier. Upload it again before publishing."
    if not protocol_name:
        return False, "Save a protocol name in Step 1B before publishing."
    reason = publish_block_reason(
        guide,
        st.session_state.get("pipeline_result"),
    )
    if reason:
        return False, reason

    published = dict(st.session_state.get("published_protocols", {}))
    same_study = _same_study_ids()
    if any(
        other_id != protocol_id
        and other_id not in same_study
        and isinstance(record, Mapping)
        and str(record.get("name", "")).casefold() == protocol_name.casefold()
        for other_id, record in published.items()
    ):
        return False, (
            "Another published protocol already uses this name. Choose a "
            "different protocol name in Step 1B."
        )
    snapshot = _publishable(guide)
    snapshot.trial_title = protocol_name
    snapshot.protocol_id = None
    try:
        missing, fingerprint = acknowledgment_context(
            guide, st.session_state.get("pipeline_result"), protocol_id
        )
        review_store.publish(
            protocol_id, protocol_name, snapshot,
            missing_categories=missing, acknowledgment_fingerprint=fingerprint,
        )
        review_store.log_action(
            protocol_id, "published", f"{len(snapshot.all_items())} statement(s)"
        )
    except review_store.AcknowledgmentError as exc:
        return False, str(exc)
    except (sqlite3.Error, OSError, version_store.VersionError) as exc:
        return False, f"Publishing failed; nothing was changed for patients: {exc}"
    version = version_store.get_version(protocol_id) or {}
    for older in same_study:
        published.pop(older, None)  # superseded in the database by this publication
    published[protocol_id] = {
        "name": protocol_name,
        "guide": snapshot,
        "version_label": version.get("version_label"),
        "published_at": version.get("published_at"),
    }
    st.session_state["published_protocols"] = published
    persist_review()
    return True, PUBLISHED_MESSAGE


UNPUBLISHED_MESSAGE = (
    "Unpublished. This protocol is no longer visible to patients or the "
    "Patient Assistant."
)


def unpublish_current_protocol() -> tuple[bool, str]:
    """Withdraw the published snapshot so patients and the assistant lose it."""
    try:
        authz.require_coordinator()
    except authz.AuthorizationError as exc:
        return False, str(exc)
    protocol_id = st.session_state.get("coordinator_protocol_id")
    if not protocol_id or protocol_id not in st.session_state.get("published_protocols", {}):
        return False, "This protocol is not published."
    try:
        review_store.unpublish(protocol_id)
        review_store.log_action(protocol_id, "unpublished", "")
    except (sqlite3.Error, OSError) as exc:
        return False, f"Unpublish failed; the protocol is still published: {exc}"
    published = dict(st.session_state["published_protocols"])
    published.pop(protocol_id, None)
    st.session_state["published_protocols"] = published
    return True, UNPUBLISHED_MESSAGE


def sync_published_from_store() -> None:
    """
    Make the stored publication state the truth for patients.

    Published rows replace the session copy, withdrawn rows remove it, and
    protocols the database does not know about are left alone. This is what
    lets a restart (or another browser session) see the latest decisions.
    """
    try:
        stored = review_store.load_publications()
    except (sqlite3.Error, OSError):
        return
    published = dict(st.session_state.get("published_protocols", {}))
    changed = False
    for protocol_id, record in stored.items():
        if record["status"] == review_store.PUBLISHED:
            current = published.get(protocol_id)
            if current is None or (
                isinstance(current, Mapping)
                and is_fully_approved_snapshot(current.get("guide"))
                and not (
                    current.get("name") == record["name"]
                    and current.get("guide") == record["guide"]
                    and current.get("version_label") == record["version_label"]
                    and current.get("published_at") == record["published_at"]
                )
            ):
                published[protocol_id] = {
                    "name": record["name"],
                    "guide": record["guide"],
                    "version_label": record["version_label"],
                    "published_at": record["published_at"],
                }
                changed = True
        elif published.pop(protocol_id, None) is not None:
            changed = True
    if changed:
        st.session_state["published_protocols"] = published


def _remember_saved_names() -> None:
    """Make saved protocol names known so a duplicate name is caught."""
    try:
        names = review_store.draft_names()
    except (sqlite3.Error, OSError):
        return
    protocols = st.session_state.setdefault("coordinator_protocols", {})
    for saved_id, saved_name in names.items():
        protocols.setdefault(saved_id, {"name": saved_name})


def _show_coordinator_view(view: str) -> None:
    """Switch between dashboard, protocol list and workflow."""
    st.session_state[COORD_VIEW_KEY] = view
    step = st.session_state.get("coord_step", STEPS[0])
    record_page_visit(
        coordinator_page(
            step if step in STEPS else STEPS[0],
            st.session_state.get("coord_max_step", 0),
            view,
        )
    )


def start_new_protocol(study_id: Optional[str] = None, label: Optional[str] = None) -> None:
    """
    Open the workflow empty. Saved protocols and analysis caches are untouched.

    With a study id, the upload becomes a new version of that study.
    """
    clear_analysis_state(keep_upload=False)
    st.session_state["coord_study_id"] = study_id
    st.session_state["coord_version_label"] = label or version_store.DEFAULT_LABEL
    reset_upload_widget()
    st.session_state["_coord_goto"] = STEPS[0]
    st.session_state.pop("_draft_restored", None)
    st.session_state.pop("_open_notice", None)
    _remember_saved_names()
    _show_coordinator_view(VIEW_WORKFLOW)


def open_saved_protocol(protocol_id: str) -> bool:
    """
    Load one saved protocol into the workflow without running any analysis.

    Only stored data is loaded. When the saved analysis is missing the
    workflow opens empty with an explanation instead of reusing another
    protocol's results.
    """
    clear_analysis_state(keep_upload=False)
    reset_upload_widget()
    st.session_state.pop("_draft_restored", None)
    st.session_state.pop("_open_notice", None)
    st.session_state["coord_study_id"] = None
    st.session_state["coord_version_label"] = version_store.DEFAULT_LABEL
    _remember_saved_names()
    try:
        draft = review_store.load_draft(protocol_id)
    except (sqlite3.Error, OSError):
        draft = None
    st.session_state["_coord_goto"] = STEPS[0]
    label = ""
    if draft is not None:
        label = draft["name"] or draft["uploaded_name"] or "this protocol"
    if draft is None:
        st.session_state["_open_notice"] = (
            "error",
            "That protocol could not be read from the database.",
        )
    elif draft["pipeline"] is None:
        st.session_state["_open_notice"] = (
            "warning",
            f"The analysis for **{label}** was not saved, so its summary and "
            "statements cannot be reopened. Its saved record is unchanged. "
            "Upload the PDF again to analyze it as a new protocol.",
        )
    else:
        protocols = st.session_state.setdefault("coordinator_protocols", {})
        protocols[protocol_id] = {
            "name": draft["name"],
            "fingerprint": draft["upload_fingerprint"],
            "uploaded_name": draft["uploaded_name"],
        }
        version = version_store.get_version(protocol_id)
        if version:
            st.session_state["coord_study_id"] = version["study_id"]
            st.session_state["coord_version_label"] = version["version_label"]
        st.session_state.pipeline_result = draft["pipeline"]
        st.session_state.coordinator_protocol_id = protocol_id
        st.session_state.uploaded_name = draft["uploaded_name"]
        st.session_state.uploaded_fingerprint = draft["upload_fingerprint"]
        st.session_state[GUIDE_KEY] = draft["guide"]
        st.session_state[GUIDE_SOURCE_KEY] = analysis_fingerprint(draft["pipeline"])
        st.session_state["coord_max_step"] = len(STEPS) - 1
        st.session_state["_coord_goto"] = STEPS[2]
        st.session_state["_draft_restored"] = label
    _show_coordinator_view(VIEW_WORKFLOW)
    return draft is not None and draft["pipeline"] is not None


def _need_analysis() -> bool:
    """Show a friendly note and return True when no document is analysed yet."""
    if st.session_state.pipeline_result is not None:
        return False
    if render_analysis_status():
        return True
    if st.session_state.get("analysis_error"):
        st.error(st.session_state.analysis_error)
        return True
    st.info("Start with step 1: upload a document and click **Analyze**.")
    return True


def _go_to(step: str) -> None:
    st.session_state["_coord_goto"] = step
    record_page_visit(
        coordinator_page(
            step,
            max(
                st.session_state.get("coord_max_step", 0),
                STEPS.index(step),
            ),
        )
    )
    st.rerun()


def has_coordinator_progress() -> bool:
    """True when an analysis or guide exists that a new upload would discard."""
    return (
        st.session_state.get("pipeline_result") is not None
        or get_guide() is not None
        or active_analysis_job() is not None
    )


def adopt_upload(uploaded: Any, fingerprint: str) -> None:
    """Store a newly uploaded PDF as the current protocol document."""
    st.session_state.temp_pdf_path = persist_uploaded_pdf(uploaded)
    st.session_state.uploaded_name = uploaded.name
    st.session_state.uploaded_fingerprint = fingerprint
    st.session_state.coordinator_protocol_id = uuid.uuid4().hex
    st.session_state.coord_max_step = 0
    st.session_state.pop("_open_notice", None)
    study_id = st.session_state.get("coord_study_id")
    if study_id:
        # A new version defaults to the study's title as its protocol name.
        try:
            title = version_store.study_title(study_id)
        except (sqlite3.Error, OSError):
            title = None
        if title:
            st.session_state.setdefault("coordinator_protocols", {})[
                st.session_state.coordinator_protocol_id
            ] = {"name": title, "fingerprint": fingerprint, "uploaded_name": uploaded.name}


def _render_version_field(protocol_id: str) -> None:
    """Show (or, for a brand-new trial, choose) the version this upload becomes."""
    version = version_store.get_version(protocol_id)
    if version:
        st.caption(
            f"Clinical trial: **{version['study_title']}** · Version **{version['version_label']}**"
        )
        return
    study_id = st.session_state.get("coord_study_id")
    if study_id:
        st.caption(
            f"New version of **{version_store.study_title(study_id)}**: "
            f"Version **{st.session_state.get('coord_version_label')}**"
        )
        return
    key = f"version_input_{protocol_id}"
    st.text_input(
        "Version identifier",
        key=key,
        value=st.session_state.get("coord_version_label") or version_store.DEFAULT_LABEL,
        max_chars=20,
        help=version_store.LABEL_RULES,
    )
    st.session_state["coord_version_label"] = st.session_state[key].strip()


def reset_upload_widget() -> None:
    """Recreate the uploader empty by changing its widget key."""
    st.session_state["upload_widget_version"] = (
        st.session_state.get("upload_widget_version", 0) + 1
    )


def render_upload_step() -> None:
    """Step 1: upload a PDF, name the protocol, and run its analysis."""
    left, right = st.columns([1.25, 1], gap="large")

    with right:
        with st.container(key="doctype_card"):
            st.html('<div class="ctai-label">Step 1b</div><h3 style="margin:0 0 .4rem">Enter protocol name</h3>')
            protocol_id = st.session_state.get("coordinator_protocol_id")
            if protocol_id:
                name_key = f"protocol_name_{protocol_id}"
                st.text_input(
                    "Protocol name",
                    key=name_key,
                    value=current_protocol_name(),
                    placeholder="e.g. Type 2 Diabetes Treatment Study",
                    label_visibility="collapsed",
                )
                if st.button("Save", key="save_protocol_name_btn"):
                    protocol_name = st.session_state[name_key].strip()
                    label_problem = None
                    if not version_store.get_version(protocol_id) and not st.session_state.get("coord_study_id"):
                        try:
                            version_store.validate_label(
                                st.session_state.get("coord_version_label") or version_store.DEFAULT_LABEL
                            )
                        except version_store.VersionError as exc:
                            label_problem = str(exc)
                    if not protocol_name:
                        st.warning("Enter a protocol name before saving.")
                    elif label_problem:
                        st.warning(label_problem)
                    else:
                        protocols = st.session_state.setdefault("coordinator_protocols", {})
                        same_study = _same_study_ids()
                        duplicate = any(
                            existing_id != protocol_id
                            and existing_id not in same_study
                            and record.get("name", "").casefold() == protocol_name.casefold()
                            for existing_id, record in protocols.items()
                        )
                        if duplicate:
                            st.warning("That name is already saved for another protocol. Choose a different name.")
                        else:
                            protocols[protocol_id] = {
                                "name": protocol_name,
                                "fingerprint": st.session_state.uploaded_fingerprint,
                                "uploaded_name": st.session_state.uploaded_name,
                            }
                            persist_review()
                            st.success(f"Protocol name saved: **{protocol_name}**")
                _render_version_field(protocol_id)
                if current_protocol_name():
                    st.caption(f"Saved name: **{current_protocol_name()}**")
            else:
                st.caption("Upload a PDF to name its protocol.")

    with left:
        with st.container(key="upload_card"):
            st.html('<div class="ctai-label">Step 1a</div><h3 style="margin:0 0 .4rem">Upload your document</h3>')
            upload_key = f"protocol_upload_{st.session_state.get('upload_widget_version', 0)}"
            uploaded = st.file_uploader(
                "Document (PDF)",
                type=["pdf"],
                key=upload_key,
                label_visibility="collapsed",
                help="Select a clinical trial document as a PDF. Analysis may take several minutes.",
            )

            # The uploader widget is discarded whenever this step is not shown,
            # so the retained session data, not the widget, defines the document.
            if uploaded is not None:
                fingerprint = upload_fingerprint(uploaded)
                if fingerprint != st.session_state.uploaded_fingerprint:
                    if not has_coordinator_progress():
                        adopt_upload(uploaded, fingerprint)
                        st.rerun()
                    st.warning(
                        f"**{uploaded.name}** is different from the current protocol. "
                        "Replacing it discards the current analysis and approvals "
                        "(published protocols are not affected)."
                    )
                    replace_col, keep_col = st.columns(2)
                    if replace_col.button("Replace current protocol", key="replace_protocol_btn"):
                        clear_analysis_state(keep_upload=False)
                        adopt_upload(uploaded, fingerprint)
                        st.rerun()
                    if keep_col.button("Keep current protocol", key="keep_protocol_btn"):
                        reset_upload_widget()
                        st.rerun()

            if st.session_state.uploaded_fingerprint:
                document_missing = (
                    st.session_state.pipeline_result is None
                    and active_analysis_job() is None
                    and not os.path.isfile(st.session_state.temp_pdf_path or "")
                )
                if document_missing:
                    st.warning("The uploaded file is no longer available. Upload the PDF again.")
                    clear_analysis_state(keep_upload=False)
                elif active_analysis_job() is not None:
                    st.success(f"Analyzing: **{st.session_state.uploaded_name}**")
                elif st.session_state.pipeline_result is None:
                    st.success(f"Ready to analyze: **{st.session_state.uploaded_name}**")
                else:
                    st.success(f"Analysis cached for: **{st.session_state.uploaded_name}**")
                    st.progress(1.0, text="100% complete — Analysis complete")

                running = render_analysis_status()
                analyze_clicked = st.button(
                    "Analyze protocol",
                    type="primary",
                    key="analyze_btn",
                    disabled=st.session_state.pipeline_result is not None or running,
                )
                if analyze_clicked:
                    start_analysis()
                    st.rerun()

                if st.session_state.pop("_analysis_just_completed", False) and (
                    st.session_state.pipeline_result is not None
                ):
                    _go_to(STEPS[1])

                if st.session_state.analysis_error:
                    st.error(st.session_state.analysis_error)

                if st.session_state.pipeline_result is not None and st.button(
                    "Continue to the summary", type="primary", key="continue_summary_btn"
                ):
                    _go_to(STEPS[1])

                if st.button("Clear analysis / start over", key="clear_analysis_btn"):
                    clear_analysis_state(keep_upload=False)
                    reset_upload_widget()
                    st.rerun()

            elif st.session_state.pipeline_result is not None:
                st.caption(
                    f"Using cached analysis for **{st.session_state.uploaded_name}**. "
                    "Upload a new PDF to replace it."
                )
                if st.button("Continue to the summary", type="primary", key="continue_summary_btn"):
                    _go_to(STEPS[1])


def render_summary_step() -> None:
    """Step 2: what the AI found in the document."""
    if _need_analysis():
        return
    result = st.session_state.pipeline_result
    protocol_name = current_protocol_name()
    st.html(
        '<div class="ctai-chip-row" style="justify-content:flex-start;margin-bottom:.6rem">'
        f'{theme.chip(protocol_name or "Protocol name not saved", "lav")}'
        f'{theme.chip(st.session_state.uploaded_name or "Cached analysis", "grey")}</div>'
    )
    st.caption(
        "This is what the AI found, with the page each item came from. "
        "Next, it turns these into plain-language statements for you to check."
    )
    render_analysis_completeness(result)
    relevance = result.get("relevance") or {}
    if relevance.get("skipped_pages"):
        pages = ", ".join(str(p["page"]) for p in relevance["skipped_pages"])
        st.caption(
            f"Pages not sent to the AI because every section looked purely administrative: {pages}. "
            "The full protocol is still used to check page citations."
        )
    audit = result.get("completeness_audit") or {}
    if audit.get("needs_attention"):
        items = [*(f"No {m} found" for m in audit.get("missing", [])),
                 *audit.get("uncertain", []), *audit.get("notes", [])]
        st.caption("Worth double-checking against the protocol: " + "; ".join(items) + ".")
    render_protocol_summary(result)
    protocol_id = st.session_state.get("coordinator_protocol_id")
    if protocol_id:
        render_summary_generation(result, protocol_id)
    if st.button("Continue to review statements", type="primary", key="continue_review_btn"):
        _go_to(STEPS[2])


def render_review_step() -> None:
    """
    Step 3 — four sections: review statements, add a statement, patient
    summary (collapsed), publish.
    """
    if _need_analysis():
        return
    protocol_name = current_protocol_name()
    if protocol_name:
        st.caption(f"Protocol: **{protocol_name}**")
    if not render_guide_review_tab(st.session_state.pipeline_result) or get_guide() is None:
        # Statements are still being generated (or failed): nothing to publish yet.
        return

    protocol_id = st.session_state.get("coordinator_protocol_id")
    if protocol_id:
        render_summary_panel(get_guide(), protocol_id)
        render_version_panel(get_guide(), protocol_id)

    guide = get_guide()
    publish_reason = (
        "Save a protocol name in Step 1B before publishing."
        if not protocol_name
        else publish_block_reason(guide, st.session_state.get("pipeline_result"))
    )

    restored = st.session_state.pop("_draft_restored", None)
    if restored:
        st.info(f"Restored your saved review for **{restored}**.")

    record = published_record(protocol_id)
    st.divider()
    _render_missing_categories(guide, protocol_id, version_label_of(protocol_id))
    publish_clicked = st.button(
        "Publish to Patient Portal",
        type="primary",
        key="publish_protocol_btn",
        disabled=publish_reason is not None,
        help=publish_reason,
        use_container_width=True,
    )
    # Only this run's attempt is reported; nothing is carried over between runs.
    attempt = publish_current_protocol() if publish_clicked else None
    if attempt is not None and attempt[0]:
        record = published_record(protocol_id)
    _render_publish_status(guide, protocol_id, protocol_name, publish_reason, record, attempt)

    # Only for a protocol that is really published.
    if record is not None and st.button(
        "Preview as a patient", key="preview_patient_btn"
    ):
        navigate_to_page(patient_page(protocol_id))
        st.rerun()


ACK_TEXT = (
    "I have reviewed the missing categories, understand the risks, and accept "
    "responsibility for publishing this version with incomplete information."
)
ACK_KEY = "publish_ack_checkbox"


def version_label_of(protocol_id: Optional[str]) -> Optional[str]:
    return (version_store.get_version(protocol_id) or {}).get("version_label") if protocol_id else None


def _set_acknowledgment(protocol_id: str, missing: list[str], fingerprint: str, version: Optional[str]) -> None:
    """Checkbox callback: store or withdraw the acknowledgment server-side."""
    try:
        if st.session_state.get(ACK_KEY):
            review_store.record_incomplete_acknowledgment(protocol_id, fingerprint, missing, version)
        else:
            review_store.clear_incomplete_acknowledgment(protocol_id)
    except (authz.AuthorizationError, sqlite3.Error, OSError):
        st.session_state[ACK_KEY] = False


def _render_missing_categories(guide: Any, protocol_id: Optional[str], version: Optional[str]) -> None:
    """Warning and acknowledgment for critical categories with no approved statement."""
    pipeline_result = st.session_state.get("pipeline_result")
    if not protocol_id or pipeline_result is None:
        return
    try:
        missing, fingerprint = acknowledgment_context(guide, pipeline_result, protocol_id)
        acknowledged = review_store.acknowledgment_valid(protocol_id, fingerprint)
    except (sqlite3.Error, OSError):
        return
    if not missing:
        st.session_state.pop(ACK_KEY, None)
        return
    st.warning(
        "**These required categories have no approved statement:** "
        + ", ".join(missing)
        + ". You can publish anyway, but patients will not see this information."
    )
    # Sync the checkbox from the stored acknowledgment before it is drawn so
    # a valid ack survives Streamlit reruns, and an invalidated one clears.
    if st.session_state.get(ACK_KEY) != acknowledged:
        st.session_state[ACK_KEY] = acknowledged
    st.checkbox(
        ACK_TEXT,
        key=ACK_KEY,
        on_change=_set_acknowledgment,
        args=(protocol_id, missing, fingerprint, version),
    )


def _render_publish_status(
    guide: Any,
    protocol_id: Optional[str],
    protocol_name: str,
    publish_reason: Optional[str],
    record: Optional[Mapping[str, Any]],
    attempt: Optional[tuple],
) -> None:
    """Publication status, blockers, and Unpublish (below the sticky bar)."""
    status = "Not published"
    if record is not None:
        status = (
            "Published"
            if published_matches_current(record, guide, protocol_name)
            else "Published (earlier version; newer changes not published)"
        )
    elif review_store.publication_status(protocol_id) == review_store.WITHDRAWN:
        status = "Unpublished (withdrawn from patients)"
    st.caption(f"Publication status: **{status}**")

    if attempt is not None and not attempt[0]:
        st.error(attempt[1])
    elif record is not None and published_matches_current(record, guide, protocol_name):
        st.success(PUBLISHED_MESSAGE)
    elif record is not None:
        st.info(
            "An earlier version of this protocol is published. Your latest "
            "changes are not published yet."
        )
        if publish_reason:
            st.warning(publish_reason)
    elif publish_reason:
        st.warning(publish_reason)

    unpublished_note = st.session_state.pop("_unpublish_message", None)
    if record is not None and st.button(
        "Unpublish (withdraw from patients)", key="unpublish_protocol_btn"
    ):
        ok, message = unpublish_current_protocol()
        st.session_state["_unpublish_message"] = (ok, message)
        st.rerun()
    if unpublished_note:
        (st.success if unpublished_note[0] else st.error)(unpublished_note[1])


def unsaved_work() -> tuple[list[str], list[str]]:
    """
    What the coordinator would lose by leaving the current protocol.

    Returns (typed, other). `typed` is text typed into a box but not saved;
    it disappears as soon as that page is left. `other` is work that stays in
    the session but is not stored as a protocol (a running analysis, statements
    being generated, an analysis that has no saved review yet, or an upload that
    has not been analyzed). Saved protocols are never counted.
    """
    typed: list[str] = []
    other: list[str] = []
    state = st.session_state
    protocol_id = state.get("coordinator_protocol_id")

    guide = get_guide()
    if guide is not None:
        edits = sum(
            1
            for item in guide.all_items()
            if isinstance(state.get(f"guide_{item.item_id}_text"), str)
            and state[f"guide_{item.item_id}_text"].strip() != item.plain_text
        )
        if edits:
            typed.append(f"{edits} statement edit(s) you have not saved")
    if protocol_id:
        name_value = state.get(f"protocol_name_{protocol_id}")
        if isinstance(name_value, str) and name_value.strip() != current_protocol_name():
            typed.append("a protocol name you have not saved")
        draft_text = state.get(f"summary_unsaved_{protocol_id}")
        if isinstance(draft_text, str):
            try:
                saved = review_store.load_summary(protocol_id)
            except (sqlite3.Error, OSError):
                saved = None
            if saved is None or draft_text.strip() != saved["draft_text"].strip():
                typed.append("summary edits you have not saved")

    if active_analysis_job() is not None:
        other.append("an analysis is still running")
    generating = guide_jobs.jobs.get(state.get(GUIDE_JOB_KEY))
    if generating is not None and generating.active:
        other.append("patient statements are still being generated")
    if state.get("pipeline_result") is not None and protocol_id:
        try:
            saved_review = review_store.draft_exists(protocol_id)
        except (sqlite3.Error, OSError):
            saved_review = True
        if not saved_review:
            other.append(
                "the analysis has not been saved as a protocol yet "
                "(open the Review step to save it)"
            )
    elif (
        state.get("uploaded_fingerprint")
        and state.get("pipeline_result") is None
        and active_analysis_job() is None
    ):
        other.append("a PDF is uploaded but has not been analyzed")
    return typed, other


def _typed_widget_values() -> dict[str, str]:
    """Text boxes whose typed value differs from what is saved (for 'Stay')."""
    state = st.session_state
    values: dict[str, str] = {}
    guide = get_guide()
    for item in guide.all_items() if guide is not None else []:
        key = f"guide_{item.item_id}_text"
        if isinstance(state.get(key), str) and state[key].strip() != item.plain_text:
            values[key] = state[key]
    if state.get("coord_step") in STEPS:
        values["coord_step"] = state["coord_step"]
    protocol_id = state.get("coordinator_protocol_id")
    name_key = f"protocol_name_{protocol_id}"
    if protocol_id and isinstance(state.get(name_key), str):
        values[name_key] = state[name_key]
    return values


def _discard_note(typed: list[str], other: list[str]) -> str:
    items = typed + other
    return "; ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]


def _leave_workflow_for_home() -> None:
    """Return to Home. Typed-but-unsaved text is dropped; saved data is untouched."""
    st.session_state.pop("_nav_pending", None)
    for key in [k for k in st.session_state if str(k).startswith("summary_unsaved_")]:
        del st.session_state[key]
    _show_coordinator_view(VIEW_HOME)


def _request_nav(kind: str, protocol_id: str = "", label: str = "") -> None:
    """Run a navigation now, or ask first when it would lose unsaved work."""
    typed, other = unsaved_work()
    if kind == "back":
        needs_confirm = bool(typed)
        other = []
    else:
        needs_confirm = bool(typed or other)
    if needs_confirm:
        st.session_state["_nav_pending"] = {
            "kind": kind,
            "protocol_id": protocol_id,
            "label": label,
            "typed": typed,
            "other": other,
            "restore": _typed_widget_values() if kind == "back" else {},
        }
    elif kind == "back":
        _leave_workflow_for_home()
    else:
        _run_switch(kind, protocol_id, label)
    st.rerun()


def _run_switch(kind: str, protocol_id: str, label: str = "") -> None:
    st.session_state.pop("_nav_pending", None)
    if kind == "add":
        start_new_protocol()
    elif kind == "version":
        start_new_protocol(protocol_id, label)
    else:
        open_saved_protocol(protocol_id)


def render_nav_confirmation() -> bool:
    """Show the pending 'discard unsaved work?' prompt. True when it was shown."""
    pending = st.session_state.get("_nav_pending")
    if not pending:
        return False
    render_top_bar(COORDINATOR)
    what = _discard_note(pending["typed"], pending["other"])
    if pending["kind"] == "back":
        st.warning(
            f"You have {what}. Going back to the dashboard discards it. "
            "Anything already saved is kept."
        )
        go_label, stay_label = "Discard it and go back", "Stay on this page"
    else:
        target = {
            "add": "start a new protocol",
            "version": "start a new version",
        }.get(pending["kind"], "open another protocol")
        st.warning(
            f"Before you {target}: {what}. Continuing discards the current "
            "working session. Saved protocols and analysis caches are not affected."
        )
        go_label, stay_label = "Discard it and continue", "Keep it running"
    go, stay = st.columns(2)
    if go.button(go_label, key="nav_confirm_btn"):
        if pending["kind"] == "back":
            _leave_workflow_for_home()
        else:
            _run_switch(pending["kind"], pending["protocol_id"], pending.get("label", ""))
        st.rerun()
    if stay.button(stay_label, key="nav_cancel_btn"):
        st.session_state.pop("_nav_pending", None)
        if pending["kind"] != "back":
            _show_coordinator_view(VIEW_HOME)
        # Text boxes were not drawn while the prompt showed, so put the typed text back.
        for key, value in pending.get("restore", {}).items():
            st.session_state[key] = value
        st.rerun()
    return True


def render_coordinator_dashboard() -> None:
    """Home: add a protocol or open a saved one. Shows no analysis or review data."""
    _show_coordinator_view(VIEW_HOME)
    render_top_bar(COORDINATOR)
    typed, other = unsaved_work()
    resumable = None
    if other:
        resumable = st.session_state.get("uploaded_name") or "your upload"
    action = coordinator_home.render_home(
        st.session_state.get(NAME_KEY) or "", resumable, _discard_note([], other) if other else ""
    )
    if action == coordinator_home.ACTION_RESUME:
        _show_coordinator_view(VIEW_WORKFLOW)
        st.rerun()
    elif action == coordinator_home.ACTION_ADD:
        _request_nav("add")
    elif action == coordinator_home.ACTION_VERSION:
        _show_coordinator_view(VIEW_VERSIONS)
        st.rerun()
    elif action == coordinator_home.ACTION_VIEW:
        _show_coordinator_view(VIEW_LIST)
        st.rerun()


def render_protocol_list_view() -> None:
    render_top_bar(COORDINATOR)
    _back_to_dashboard_button("list_back_btn", leave=False)
    chosen = coordinator_home.render_protocol_list()
    if chosen is None:
        return
    kind, protocol_id = chosen
    _request_nav(kind, protocol_id)


def render_versions_view() -> None:
    render_top_bar(COORDINATOR)
    _back_to_dashboard_button("versions_back_btn", leave=False)
    chosen = coordinator_home.render_version_picker()
    if chosen is None:
        return
    kind, payload = chosen
    if kind == "version":
        study_id, _, label = payload.partition("|")
        _request_nav("version", study_id, label)
    else:
        _request_nav(kind)


def _back_to_dashboard_button(key: str, *, leave: bool = True) -> None:
    if st.button("Back to Dashboard", key=key, icon=":material/arrow_back:"):
        if leave:
            _request_nav("back")
        else:
            _show_coordinator_view(VIEW_HOME)
            st.rerun()


def render_coordinator_workspace() -> None:
    try:
        authz.require_coordinator()
    except authz.AuthorizationError as exc:
        st.error(str(exc))
        return
    if render_nav_confirmation():
        return
    view = st.session_state.get(COORD_VIEW_KEY)
    if view == VIEW_LIST:
        render_protocol_list_view()
        return
    if view == VIEW_VERSIONS:
        render_versions_view()
        return
    if view != VIEW_WORKFLOW:
        render_coordinator_dashboard()
        return
    goto = st.session_state.pop("_coord_goto", None)
    if goto in STEPS:
        st.session_state["coord_step"] = goto
        st.session_state["coord_max_step"] = max(
            st.session_state.get("coord_max_step", 0),
            STEPS.index(goto),
        )
    st.session_state.setdefault("coord_step", STEPS[0])
    st.session_state.setdefault("coord_max_step", 0)

    current_step = st.session_state["coord_step"]
    if current_step not in STEPS:
        current_step = STEPS[0]
    current_index = STEPS.index(current_step)
    max_step = min(st.session_state["coord_max_step"], len(STEPS) - 1)
    if current_index > max_step:
        current_step = STEPS[max_step]
        st.session_state["coord_step"] = current_step
    st.session_state["coord_max_step"] = max(
        st.session_state["coord_max_step"],
        current_index,
    )
    record_page_visit(
        coordinator_page(current_step, st.session_state["coord_max_step"])
    )
    render_top_bar(COORDINATOR)
    _back_to_dashboard_button("workflow_back_btn")

    name = st.session_state.get(NAME_KEY) or ""
    theme.page_head(
        f"Welcome, {name}" if name else "Coordinator workspace",
        "Upload a document, check what the AI found, then approve what patients will read.",
    )
    notice = st.session_state.get("_open_notice")
    if notice:
        (st.error if notice[0] == "error" else st.warning)(notice[1])

    step = st.segmented_control(
        "Step",
        STEPS[: st.session_state["coord_max_step"] + 1],
        key="coord_step",
        required=True,
        label_visibility="collapsed",
    ) or STEPS[0]

    st.write("")
    if step == STEPS[0]:
        render_upload_step()
    elif step == STEPS[1]:
        render_summary_step()
    elif step == STEPS[2]:
        render_review_step()


def allowed_roles(account_role: str) -> tuple[str, ...]:
    """Workspaces an account may open; coordinators may preview the patient portal."""
    return (COORDINATOR, PATIENT) if account_role == COORDINATOR else (PATIENT,)


def main() -> None:
    st.set_page_config(
        page_title="Clinical Trial AI",
        page_icon="📋",
        layout="wide",
    )
    init_session_state()
    sync_published_from_store()
    theme.inject_css()
    if collect_analysis_job() and st.session_state.get("_analysis_watched"):
        st.session_state["_analysis_just_completed"] = True

    user = authz.current_user()
    role = st.session_state.get(ROLE_KEY)
    # The workspace always follows the signed-in account, never a stored or navigated role.
    if user is None or role not in allowed_roles(user["role"]):
        role = None
        st.session_state[ROLE_KEY] = None
    if user is None:
        st.session_state.pop(authz.TOKEN_KEY, None)

    if role and st.session_state.get("show_profile"):
        render_top_bar(role)
        auth_pages.render_profile_page(user)
    elif role == COORDINATOR:
        render_coordinator_workspace()
    elif role == PATIENT:
        render_patient_workspace()
    else:
        render_landing()

    st.session_state.pop("_analysis_just_completed", None)
    st.session_state.pop("_analysis_watched", None)
    if role:
        st.caption(RESEARCH_DISCLAIMER)


if __name__ == "__main__":
    main()
