"""
Clinical Trial AI — Streamlit MVP.

Wires Phase 3 pipeline + Phase 4/5 assistants. Analysis runs once per
upload and is cached in session state. Does not modify backend modules.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import streamlit as st

from models.protocol_models import ProtocolAnalysis
from services.coordinator_query import answer_from_pipeline_result
from agents.guide_patient_agent import APPROVED_STATUS
from models.protocol_models import ProtocolAnalysis
from services.coordinator_query import answer_from_pipeline_result
from services.model_provider import provider_display_name
from services.protocol_pipeline import analyze_protocol_pdf
from ui import theme
from ui.landing import COORDINATOR, NAME_KEY, PATIENT, ROLE_KEY, render_landing, render_top_bar
from ui.patient_guide_tab import clear_guide, get_guide, render_guide_review_tab
from ui.patient_view import render_patient_workspace

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
        ROLE_KEY: None,
        NAME_KEY: "",
        "doc_type": DOC_TYPES[0],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_analysis_state(*, keep_upload: bool = False) -> None:
    """Drop cached analysis and Q&A history; optionally keep the upload path."""
    cleanup_temp_pdf(st.session_state.get("temp_pdf_path"))
    st.session_state.pipeline_result = None
    st.session_state.analysis_error = None
    st.session_state.coordinator_history = []
    st.session_state.patient_history = []
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
    return f"{name}:{size}"


def run_protocol_analysis(pdf_path: str) -> dict[str, Any]:
    """
    Call the Phase 3 pipeline. Separated for mocking in tests.

    Raises ValueError for empty/invalid extractable content and may raise
    connection/runtime errors if the configured model provider is unavailable.
    """
    return analyze_protocol_pdf(pdf_path)


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
    if not status:
        return "unknown"
    return status


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
        st.subheader("Needs review")
        st.warning(
            f"{len(attention)} item(s) could not be fully verified. "
            "Check them against the protocol before approving patient content.\n\n"
            + "\n".join(f"- {w}" for w in attention)
        )
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
                f"  _Source page: {page_label} · Verification: `{status}`_"
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
                f"  _Source page: {page} · Verification: `{status}`_"
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


def _render_tab_disclaimers(disclaimers: Sequence[str]) -> None:
    """Show assistant disclaimers once per tab in a compact expander."""
    items = [item for item in disclaimers if item]
    with st.expander("Important disclaimers", expanded=False):
        if not items:
            st.caption(RESEARCH_DISCLAIMER)
            return
        for item in items:
            st.markdown(f"- {item}")
        st.caption("Also see the research-prototype notice at the top of the page.")


def _render_answer_block(
    answer: Any,
    *,
    audience: Optional[str] = None,
    show_disclaimers: bool = False,
) -> None:
    """Render a cleaned answer with citations/metadata in expanders."""
    intent = getattr(answer, "intent", "unknown")
    confidence = getattr(answer, "confidence", "unknown")
    citations = list(getattr(answer, "citations", []) or [])
    limitations = filter_user_facing_warnings(
        list(getattr(answer, "limitations", []) or [])
    )
    disclaimers = list(getattr(answer, "disclaimers", []) or [])
    resolved_audience = audience or getattr(answer, "audience", None)

    meta = f"**Intent:** `{intent}` · **Confidence:** `{confidence}`"
    if resolved_audience:
        meta += f" · **Audience:** `{resolved_audience}`"
    st.markdown(meta)

    display_text = format_display_answer_text(
        answer,
        audience=resolved_audience,
    )
    st.markdown(display_text)

    if citations:
        with st.expander("Citations and verification details", expanded=False):
            for citation in citations:
                page = (
                    citation.source_page
                    if citation.source_page is not None
                    else "unassigned"
                )
                st.markdown(
                    f"- **Statement:** {citation.statement}  \n"
                    f"  Category: `{citation.category}` · "
                    f"Source page: `{page}` · "
                    f"Verification: `{citation.verification_status}`"
                )
                plain = getattr(citation, "plain_language", None)
                if plain and plain.strip() and plain.strip() != citation.statement:
                    st.caption(f"Plain-language framing: {plain}")

    if limitations:
        with st.expander("Limitations", expanded=False):
            for item in limitations:
                st.markdown(f"- {item}")

    if show_disclaimers and disclaimers:
        with st.expander("Disclaimers for this answer", expanded=False):
            for item in disclaimers:
                st.markdown(f"- {item}")


def render_coordinator_tab(pipeline_result: Mapping[str, Any]) -> None:
    """Coordinator Q&A over the cached pipeline result."""
    st.caption(
        "Ask about inclusion/exclusion criteria, visits, procedures, safety, "
        "or study metadata. Answers use verified extracted facts only."
    )
    _render_tab_disclaimers(
        [
            "Coordinator answers are for research-staff review of extracted "
            "protocol facts only.",
            "This assistant cannot determine patient eligibility or provide "
            "medical advice.",
        ]
    )
    query = st.text_input(
        "Coordinator question",
        key="coordinator_query_input",
        placeholder="e.g. What are the inclusion criteria?",
    )
    ask = st.button("Ask coordinator assistant", key="coordinator_ask_btn")

    if ask:
        if not query.strip():
            st.warning("Enter a question before asking.")
        else:
            answer = answer_from_pipeline_result(query.strip(), pipeline_result)
            st.session_state.coordinator_history.append(
                {"query": query.strip(), "answer": answer}
            )

    for item in reversed(st.session_state.coordinator_history):
        st.markdown(f"#### Q: {item['query']}")
        _render_answer_block(item["answer"])
        st.divider()


DOC_TYPES = [
    "Clinical trial protocol",
    "Informed consent form",
    "Investigator's brochure",
    "Other study document",
]
STEPS = ["1 · Upload", "2 · Summary", "3 · Review", "4 · Publish"]


def _need_analysis() -> bool:
    """Show a friendly note and return True when no document is analysed yet."""
    if st.session_state.pipeline_result is not None:
        return False
    st.info("Start with step 1: upload a document and click **Analyze**.")
    return True


def _go_to(step: str) -> None:
    st.session_state["_coord_goto"] = step
    st.rerun()


def render_upload_step() -> None:
    """Step 1: upload a PDF, say what kind of document it is, run the analysis."""
    left, right = st.columns([1.25, 1], gap="large")

    with right:
        with st.container(key="doctype_card"):
            st.html('<div class="ctai-label">Step 1b</div><h3 style="margin:0 0 .4rem">What type of document is this?</h3>')
            doc_type = st.selectbox(
                "Document type",
                DOC_TYPES,
                key="doc_type",
                label_visibility="collapsed",
            )
            if doc_type == DOC_TYPES[0]:
                st.success(
                    "Best supported. Clinical Trial AI was built and tested on "
                    "trial protocols."
                )
            else:
                st.warning(
                    "This prototype is tuned for trial protocols. Other documents "
                    "can be analysed, but expect more items to need your review."
                )

    with left:
        with st.container(key="upload_card"):
            st.html('<div class="ctai-label">Step 1a</div><h3 style="margin:0 0 .4rem">Upload your document</h3>')
            uploaded = st.file_uploader(
                "Document (PDF)",
                type=["pdf"],
                label_visibility="collapsed",
                help="Select a clinical trial document as a PDF. Analysis may take several minutes.",
            )

            if uploaded is not None:
                fingerprint = upload_fingerprint(uploaded)
                if fingerprint != st.session_state.uploaded_fingerprint:
                    clear_analysis_state(keep_upload=False)
                    temp_path = persist_uploaded_pdf(uploaded)
                    st.session_state.temp_pdf_path = temp_path
                    st.session_state.uploaded_name = uploaded.name
                    st.session_state.uploaded_fingerprint = fingerprint

                st.success(f"Ready to analyze: **{st.session_state.uploaded_name}**")

                analyze_clicked = st.button(
                    "Analyze protocol",
                    type="primary",
                    key="analyze_btn",
                    disabled=st.session_state.pipeline_result is not None,
                )

                if analyze_clicked and st.session_state.pipeline_result is None:
                    st.session_state.analysis_error = None
                    progress = st.progress(0, text="Preparing analysis…")
                    status = st.empty()
                    try:
                        progress.progress(15, text="Saving upload and starting pipeline…")
                        status.info(
                            "Running PDF extraction, focused protocol analysis, "
                            "accuracy checks, and source verification. This can take "
                            f"several minutes with {provider_display_name()}."
                        )
                        progress.progress(
                            40,
                            text=f"Analyzing document ({provider_display_name()})…",
                        )
                        result = run_protocol_analysis(st.session_state.temp_pdf_path)
                        complete, status_message = analysis_completeness_message(result)
                        progress.progress(
                            100,
                            text=(
                                "Analysis complete."
                                if complete
                                else "Analysis incomplete — review warning."
                            ),
                        )
                        st.session_state.pipeline_result = result
                        st.session_state.analysis_error = None
                        if complete:
                            status.success(
                                status_message
                                + " Asking questions will not re-run PDF analysis."
                            )
                        else:
                            status.warning(status_message)
                    except Exception as exc:  # noqa: BLE001 — surface all analyze failures
                        st.session_state.pipeline_result = None
                        st.session_state.analysis_error = format_analysis_error(exc)
                        progress.progress(0, text="Analysis failed.")
                        status.empty()
                    else:
                        _go_to(STEPS[1])

                if st.session_state.analysis_error:
                    st.error(st.session_state.analysis_error)

                if st.button("Clear analysis / start over", key="clear_analysis_btn"):
                    clear_analysis_state(keep_upload=False)
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
    st.html(
        '<div class="ctai-chip-row" style="justify-content:flex-start;margin-bottom:.6rem">'
        f'{theme.chip(st.session_state.doc_type, "lav")}'
        f'{theme.chip(st.session_state.uploaded_name or "Cached analysis", "grey")}</div>'
    )
    st.caption(
        "This is what the AI found, with the page each item came from. "
        "Next, it turns these into plain-language statements for you to check."
    )
    render_analysis_completeness(result)
    render_protocol_summary(result)
    if st.button("Continue to review statements", type="primary", key="continue_review_btn"):
        _go_to(STEPS[2])


def render_review_step() -> None:
    """Step 3: mark each generated statement accurate or not, then approve."""
    if _need_analysis():
        return
    render_guide_review_tab(st.session_state.pipeline_result)


def render_publish_step() -> None:
    """Step 4: see what patients can read, preview it, ask coordinator questions."""
    if _need_analysis():
        return
    guide = get_guide()
    if guide is None:
        st.info("Generate and review the statements in step 3 first. Nothing is published yet.")
        return

    total = len(guide.all_items())
    approved = guide.approved_count()
    theme.metrics_row([
        ("Visible to patients", f"{approved} / {total}", False),
        ("Still hidden", total - approved, total - approved > 0),
        ("Rejected", guide.rejected_count(), guide.rejected_count() > 0),
        ("Document type", st.session_state.doc_type.split(" ")[0], False),
    ])
    theme.progress_bar(approved / total if total else 0)
    if approved == 0:
        st.warning("Patients cannot see anything yet. Approve at least one statement in step 3.")
    elif approved < total:
        st.info("Patients can see the approved statements. The others stay hidden until you approve them.")
    else:
        st.success("Everything is approved and visible to patients.")

    c1, c2, _ = st.columns([1.4, 1.4, 2])
    if c1.button("Preview as a patient", type="primary", key="preview_patient_btn"):
        st.session_state[ROLE_KEY] = PATIENT
        st.rerun()
    if c2.button("Back to review", key="back_review_btn"):
        _go_to(STEPS[2])

    st.html('<div class="ctai-section-title" style="text-align:left;margin-top:1.6rem"><p>For you</p><h2>Ask about the protocol</h2></div>')
    render_coordinator_tab(st.session_state.pipeline_result)


def render_coordinator_workspace() -> None:
    render_top_bar(COORDINATOR)

    name = st.session_state.get(NAME_KEY) or ""
    theme.page_head(
        f"Welcome, {name}" if name else "Coordinator workspace",
        "Upload a document, check what the AI found, then approve what patients will read.",
    )

    # Move to a step requested by a button on the previous run (a widget's
    # value can only be changed before the widget is drawn).
    goto = st.session_state.pop("_coord_goto", None)
    if goto in STEPS:
        st.session_state["coord_step"] = goto
    st.session_state.setdefault("coord_step", STEPS[0])

    step = st.segmented_control(
        "Step", STEPS, key="coord_step", required=True, label_visibility="collapsed"
    ) or STEPS[0]

    st.write("")
    if step == STEPS[0]:
        render_upload_step()
    elif step == STEPS[1]:
        render_summary_step()
    elif step == STEPS[2]:
        render_review_step()
    else:
        render_publish_step()


def main() -> None:
    st.set_page_config(
        page_title="Clinical Trial AI",
        page_icon="📋",
        layout="wide",
    )
    init_session_state()
    theme.inject_css()

    role = st.session_state.get(ROLE_KEY)
    if role == COORDINATOR:
        render_coordinator_workspace()
    elif role == PATIENT:
        render_patient_workspace()
    else:
        render_landing()

    if role:
        st.caption(RESEARCH_DISCLAIMER)


if __name__ == "__main__":
    main()
