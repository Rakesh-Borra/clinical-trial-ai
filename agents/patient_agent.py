
"""
Deterministic Patient Assistant.

Explains verified protocol facts in plain language for patients.
Does not call an LLM, invent clinical content, give medical advice,
or decide whether a specific person is eligible.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from models.patient_models import PatientAnswer, PatientCitation
from models.protocol_models import ProtocolAnalysis
from services.source_verifier import normalize_text

_DEFAULT_DISCLAIMERS = [
    (
        "This assistant explains general information from a clinical trial "
        "protocol summary for educational purposes only. It is not medical "
        "advice."
    ),
    (
        "It cannot decide whether you are eligible for a study. Talk with "
        "the study team or a qualified clinician about your personal situation."
    ),
]

_VERIFIED_STATUS = "exact_match"
_INFORMATIONAL_WARNING_MARKERS = (
    "rescue skipped",
    "criteria appear complete",
)

_ELIGIBILITY_DECISION_PATTERNS = (
    r"\bam i eligible\b",
    r"\bcan i (join|enroll|participate)\b",
    r"\bdo i qualify\b",
    r"\bis this patient eligible\b",
    r"\bcan (they|the patient|this patient) enroll\b",
    r"\bdetermine (my )?eligibility\b",
    r"\beligibility decision\b",
    # Added after safety evaluation (evaluation/run_safety_eval.py)
    r"\bgood fit\b",
    r"\b(would|will) i be (accepted|allowed|eligible)\b",
    r"\blet me (in|join|enroll|participate|take part)\b",
    r"\bcan i (be in|take part|sign up)\b",
    r"\bam i (allowed|able) to (join|enroll|participate|take part)\b",
)

_MEDICAL_ADVICE_PATTERNS = (
    r"\bdiagnose\b",
    r"\btreat(?:ment)? advice\b",
    r"\bmedical advice\b",
    r"\bshould i take\b",
    r"\bwhat medication\b",
    r"\bprescribe\b",
    # Added after safety evaluation (evaluation/run_safety_eval.py)
    r"\bshould i (stop|start|skip|change|increase|decrease|keep taking)\b",
    r"\b(okay|ok|safe|fine|alright) (for me )?to (skip|stop|take|change|miss)\b",
    r"\b(skip|stop|quit) (taking )?my\b",
    r"\bwhat dose\b",
    # Skipping, refusing or changing a study procedure is the patient's
    # personal medical decision, however the sentence is phrased.
    r"\b(skip|miss|avoid|refuse|decline|delay|postpone|stop|change|cancel)\b"
    r".{0,25}\b(blood|test|tests|testing|procedure|procedures|visit|visits|scan|exam|sample)\b",
    r"\b(do i have to|must i|am i required to|can i opt out of)\b.{0,25}\b(blood|test|procedure|scan|sample)",
)

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "study_overview": (
        "what is this study",
        "about the study",
        "study about",
        "tell me about the trial",
        "protocol title",
        "study name",
        "trial name",
        "protocol id",
    ),
    "eligibility_inclusion": (
        "inclusion",
        "who can join",
        "who may join",
        "who can participate",
        "requirements to join",
        "eligibility criteria",
    ),
    "eligibility_exclusion": (
        "exclusion",
        "who cannot join",
        "who can't join",
        "who may not join",
        "not allowed to join",
    ),
    "visits": (
        "visit",
        "schedule",
        "how often",
        "day 0",
        "day 30",
        "appointments",
        "appointment",
        "how long",
        "follow-up",
        "screening visit",
    ),
    "procedures": (
        "procedure",
        "procedures",
        "what will happen",
        "testing",
        "tests",
        "blood glucose",
        "blood pressure",
    ),
    "safety": (
        "safety",
        "side effect",
        "risk",
        "adverse",
        "hypoglycemia",
        "monitor",
        "is it safe",
        "get hurt",
        "harm",
        "dangerous",
    ),
}


def _matches_any(query: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_text(query)
    return any(re.search(pattern, normalized) for pattern in patterns)


def classify_patient_intent(query: str) -> str:
    """Classify patient intent with deterministic keyword rules."""
    if _matches_any(query, _MEDICAL_ADVICE_PATTERNS):
        return "medical_advice_refused"
    if _matches_any(query, _ELIGIBILITY_DECISION_PATTERNS):
        return "eligibility_decision_refused"

    normalized = normalize_text(query)
    scored: list[tuple[int, str]] = []
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score:
            scored.append((score, intent))

    if not scored:
        return "unsupported"

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _verification_index(
    verification: Sequence[dict],
) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for item in verification:
        category = item.get("category")
        statement = item.get("statement")
        status = item.get("status")
        if not category or statement is None or status is None:
            continue
        index[(category, normalize_text(str(statement)))] = str(status)
    return index


def _status_for(
    index: dict[tuple[str, str], str],
    category: str,
    statement: str,
) -> str:
    return index.get(
        (category, normalize_text(statement)),
        "missing_page",
    )


_VISIT_COUNT_PATTERNS = (
    r"\bhow many visits\b",
    r"\bhow many study visits\b",
    r"\bhow many (scheduled |study )?visits are there\b",
    r"\bnumber of (scheduled |study )?visits\b",
    r"\bvisit count\b",
    r"\bcount of (the )?visits\b",
)


def is_visit_count_query(query: str) -> bool:
    """True when the user is asking for a visit count, not only the schedule."""
    return _matches_any(query, _VISIT_COUNT_PATTERNS)


def distinct_scheduled_visit_names(analysis: ProtocolAnalysis) -> list[str]:
    """Return distinct visit names from analysis, preserving first-seen order."""
    seen: set[str] = set()
    names: list[str] = []
    for visit in analysis.visits:
        key = normalize_text(visit.visit_name)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(visit.visit_name)
    return names


def _distinct_visits_fully_verified(
    analysis: ProtocolAnalysis,
    verification: Sequence[dict],
) -> bool:
    names = distinct_scheduled_visit_names(analysis)
    if not names:
        return False
    index = _verification_index(verification)
    return all(
        _status_for(index, "visits", name) == _VERIFIED_STATUS for name in names
    )


def build_visit_count_preface(
    analysis: ProtocolAnalysis,
    verification: Sequence[dict],
    *,
    audience: str = "coordinator",
) -> str:
    """
    Build a visit-count lead-in.

    Confirmed numeric counts are only stated when every distinct visit is
    exact_match verified.
    """
    names = distinct_scheduled_visit_names(analysis)
    count = len(names)

    if count == 0:
        if audience == "patient":
            return (
                "A visit count is not available because no visits were "
                "present in the extracted study summary. That does not "
                "necessarily mean the full protocol lacks a schedule."
            )
        return (
            "A confirmed visit count is not available because no visits "
            "were found in the extracted protocol analysis."
        )

    if _distinct_visits_fully_verified(analysis, verification):
        visit_word = "visit" if count == 1 else "visits"
        if audience == "patient":
            return (
                f"The extracted study summary lists {count} distinct "
                f"scheduled {visit_word}. Here is the visit schedule "
                "(timing values are shown exactly as extracted):"
            )
        return (
            f"The extracted protocol analysis lists {count} distinct "
            f"scheduled {visit_word}. Verified visit schedule details:"
        )

    if audience == "patient":
        return (
            "A confirmed visit count is not available because visit "
            "information in the extracted summary is missing or not fully "
            "source-verified. Available visit schedule details:"
        )
    return (
        "A confirmed visit count is not available because one or more "
        "visit entries are missing or not fully source-verified "
        f"({_VERIFIED_STATUS}). Available visit schedule details:"
    )


def _plain_wrap(prefix: str, exact_statement: str) -> str:
    """Patient-friendly framing that keeps the exact protocol wording."""
    return f"{prefix} \"{exact_statement}\""


def _citation(
    category: str,
    statement: str,
    source_page: Optional[int],
    status: str,
    plain_language: Optional[str] = None,
) -> PatientCitation:
    return PatientCitation(
        category=category,
        statement=statement,
        source_page=source_page,
        verification_status=status,
        plain_language=plain_language,
    )


def _collect_category_citations(
    analysis: ProtocolAnalysis,
    verification: Sequence[dict],
    category: str,
) -> List[PatientCitation]:
    index = _verification_index(verification)
    citations: list[PatientCitation] = []

    if category == "inclusion_criteria":
        for item in analysis.inclusion_criteria:
            status = _status_for(index, category, item.description)
            citations.append(
                _citation(
                    category,
                    item.description,
                    item.source_page,
                    status,
                    plain_language=_plain_wrap(
                        "One requirement listed in the study summary is:",
                        item.description,
                    ),
                )
            )
    elif category == "exclusion_criteria":
        for item in analysis.exclusion_criteria:
            status = _status_for(index, category, item.description)
            citations.append(
                _citation(
                    category,
                    item.description,
                    item.source_page,
                    status,
                    plain_language=_plain_wrap(
                        "The study summary lists this as a reason someone "
                        "may not be able to join:",
                        item.description,
                    ),
                )
            )
    elif category == "procedures":
        for item in analysis.procedures:
            status = _status_for(index, category, item.name)
            statement = item.name
            if item.description:
                statement = f"{item.name} — {item.description}"
            citations.append(
                _citation(
                    category,
                    statement,
                    item.source_page,
                    status,
                    plain_language=_plain_wrap(
                        "A study activity listed in the summary is:",
                        statement,
                    ),
                )
            )
    elif category == "visits":
        for item in analysis.visits:
            status = _status_for(index, category, item.visit_name)
            timing = item.timing or "timing not stated in the summary"
            procs = (
                f"; procedures: {', '.join(item.procedures)}"
                if item.procedures
                else ""
            )
            statement = f"{item.visit_name}: {timing}{procs}"
            citations.append(
                _citation(
                    category,
                    statement,
                    item.source_page,
                    status,
                    plain_language=(
                        f"The study schedule lists {item.visit_name} at "
                        f"{timing}. Exact wording: \"{statement}\""
                    ),
                )
            )
    elif category == "safety_information":
        for item in analysis.safety_information:
            status = _status_for(index, category, item.description)
            citations.append(
                _citation(
                    category,
                    item.description,
                    item.source_page,
                    status,
                    plain_language=_plain_wrap(
                        "A safety-related statement in the study summary is:",
                        item.description,
                    ),
                )
            )
    elif category == "metadata":
        if analysis.trial_title:
            title = analysis.trial_title
            citations.append(
                _citation(
                    "metadata",
                    f"Trial title: {title}",
                    None,
                    "from_analysis",
                    plain_language=(
                        f"This study is called \"{title}\" in the summary."
                    ),
                )
            )
        if analysis.protocol_id:
            protocol_id = analysis.protocol_id
            citations.append(
                _citation(
                    "metadata",
                    f"Protocol ID: {protocol_id}",
                    None,
                    "from_analysis",
                    plain_language=(
                        f"The protocol identifier in the summary is "
                        f"\"{protocol_id}\"."
                    ),
                )
            )

    return _dedupe_citations(citations)


def _dedupe_citations(citations: Sequence[PatientCitation]) -> List[PatientCitation]:
    seen: set[tuple[str, str]] = set()
    unique: list[PatientCitation] = []
    for citation in citations:
        key = (citation.category, normalize_text(citation.statement))
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return unique


def _is_preferred_status(status: str) -> bool:
    return status in {_VERIFIED_STATUS, "from_analysis"}


def _split_verified(
    citations: Sequence[PatientCitation],
) -> tuple[list[PatientCitation], list[PatientCitation]]:
    verified = [
        c for c in citations if _is_preferred_status(c.verification_status)
    ]
    unverified = [
        c for c in citations if not _is_preferred_status(c.verification_status)
    ]
    return verified, unverified


def _format_patient_lines(citations: Iterable[PatientCitation]) -> list[str]:
    lines: list[str] = []
    for citation in citations:
        page = (
            f"source page {citation.source_page}"
            if citation.source_page is not None
            else "source page not assigned in the summary"
        )
        text = citation.plain_language or citation.statement
        lines.append(
            f"- {text} ({page}; verification: {citation.verification_status})"
        )
    return lines


def _filter_accuracy_warnings(
    accuracy_warnings: Optional[Sequence[str]],
    intent: str,
) -> list[str]:
    if not accuracy_warnings:
        return []

    relevant: list[str] = []
    for warning in accuracy_warnings:
        normalized = normalize_text(warning)
        if any(marker in normalized for marker in _INFORMATIONAL_WARNING_MARKERS):
            continue

        if intent in {
            "eligibility_inclusion",
            "eligibility_exclusion",
            "eligibility_decision_refused",
        }:
            if any(
                marker in normalized
                for marker in (
                    "hba1c",
                    "egfr",
                    "completeness",
                    "inclusion",
                    "exclusion",
                    "eligibility",
                )
            ):
                relevant.append(warning)
        elif intent == "study_overview":
            if "trial_title" in normalized or "protocol_id" in normalized:
                relevant.append(warning)
    return relevant


def _missing_summary_message(topic: str) -> str:
    return (
        f"This information about {topic} was not available in the extracted "
        "study summary. That does not necessarily mean the full protocol "
        "lacks it. Please ask the study team for the complete details. "
        "No clinical facts were invented."
    )


def _build_lookup_answer(
    *,
    intent: str,
    topic_label: str,
    preface: str,
    citations: List[PatientCitation],
    accuracy_warnings: Optional[Sequence[str]],
) -> PatientAnswer:
    verified, unverified = _split_verified(citations)
    limitations = _filter_accuracy_warnings(accuracy_warnings, intent)

    if not citations:
        return PatientAnswer(
            answer_text=_missing_summary_message(topic_label),
            intent=intent,
            citations=[],
            confidence="unavailable",
            limitations=limitations
            or [
                f"No {topic_label} were present in the extracted summary.",
            ],
            disclaimers=list(_DEFAULT_DISCLAIMERS),
            audience="patient",
        )

    if verified and not unverified:
        confidence = "high"
        body_preface = preface
        answer_citations = list(verified)
    elif verified and unverified:
        confidence = "partial"
        body_preface = (
            preface
            + " Some related items were not fully source-verified and are "
            "listed only as limitations."
        )
        limitations.append(
            f"{len(unverified)} related item(s) were not {_VERIFIED_STATUS} "
            "and are not shown as confirmed facts."
        )
        for item in unverified:
            limitations.append(
                f"Unverified ({item.verification_status}): {item.statement}"
            )
        answer_citations = list(verified)
    else:
        confidence = "partial"
        body_preface = (
            f"Some {topic_label} appeared in the summary but were not "
            f"fully source-verified ({_VERIFIED_STATUS}). Treat them as "
            "unconfirmed until reviewed with the study team:"
        )
        limitations.append(
            "No exact_match verification was available for these statements."
        )
        answer_citations = list(unverified)

    lines = _format_patient_lines(answer_citations)
    return PatientAnswer(
        answer_text=body_preface + "\n" + "\n".join(lines),
        intent=intent,
        citations=answer_citations,
        confidence=confidence,
        limitations=limitations,
        disclaimers=list(_DEFAULT_DISCLAIMERS),
        audience="patient",
    )


def _refuse_medical_advice() -> PatientAnswer:
    return PatientAnswer(
        answer_text=(
            "I cannot provide medical advice, diagnosis, or treatment "
            "recommendations. I can only explain general information from "
            "the study summary, such as visits, procedures, and listed "
            "criteria. Please talk with the study team or a qualified "
            "clinician about your health."
        ),
        intent="medical_advice_refused",
        citations=[],
        confidence="unavailable",
        limitations=[
            "Medical advice and personal clinical decisions are out of scope.",
        ],
        disclaimers=list(_DEFAULT_DISCLAIMERS),
        audience="patient",
    )


def _refuse_eligibility_decision(
    analysis: ProtocolAnalysis,
    verification: Sequence[dict],
    accuracy_warnings: Optional[Sequence[str]],
) -> PatientAnswer:
    inclusions = _collect_category_citations(
        analysis, verification, "inclusion_criteria"
    )
    exclusions = _collect_category_citations(
        analysis, verification, "exclusion_criteria"
    )
    verified_inclusions, _ = _split_verified(inclusions)
    verified_exclusions, _ = _split_verified(exclusions)

    lines = [
        "I cannot tell you whether you personally are eligible for this "
        "study. Eligibility depends on your individual situation and must "
        "be decided with the study team.",
        "Here are general criteria from the verified study summary that you "
        "can discuss with them:",
    ]
    if verified_inclusions:
        lines.append("Requirements that may allow someone to join:")
        lines.extend(_format_patient_lines(verified_inclusions))
    if verified_exclusions:
        lines.append("Reasons someone may not be able to join:")
        lines.extend(_format_patient_lines(verified_exclusions))
    if not verified_inclusions and not verified_exclusions:
        lines.append(
            "No verified inclusion or exclusion criteria were available in "
            "the extracted study summary."
        )

    citations = _dedupe_citations(
        list(verified_inclusions) + list(verified_exclusions)
    )
    limitations = [
        "This assistant does not evaluate personal eligibility.",
    ]
    limitations.extend(
        _filter_accuracy_warnings(
            accuracy_warnings,
            "eligibility_decision_refused",
        )
    )

    return PatientAnswer(
        answer_text="\n".join(lines),
        intent="eligibility_decision_refused",
        citations=citations,
        confidence="high" if citations else "unavailable",
        limitations=limitations,
        disclaimers=list(_DEFAULT_DISCLAIMERS),
        audience="patient",
    )


def _study_overview_answer(
    analysis: ProtocolAnalysis,
    verification: Sequence[dict],
    accuracy_warnings: Optional[Sequence[str]],
) -> PatientAnswer:
    citations = _collect_category_citations(
        analysis, verification, "metadata"
    )
    return _build_lookup_answer(
        intent="study_overview",
        topic_label="study name and protocol identifier details",
        preface=(
            "Here is what the extracted study summary says about this trial. "
            "Exact names and identifiers are preserved:"
        ),
        citations=citations,
        accuracy_warnings=accuracy_warnings,
    )


def answer_patient_query(
    query: str,
    analysis: ProtocolAnalysis,
    verification: list[dict],
    *,
    accuracy_warnings: Optional[list[str]] = None,
) -> PatientAnswer:
    """
    Answer a patient question using extracted analysis and verification.

    Deterministic only: no LLM calls and no invented clinical content.
    """
    if not query or not query.strip():
        return PatientAnswer(
            answer_text=(
                "Please ask a question about the study summary, such as who "
                "may join, visits, procedures, or safety information."
            ),
            intent="unsupported",
            citations=[],
            confidence="unavailable",
            limitations=["Empty query."],
            disclaimers=list(_DEFAULT_DISCLAIMERS),
            audience="patient",
        )

    intent = classify_patient_intent(query)

    if intent == "medical_advice_refused":
        return _refuse_medical_advice()

    if intent == "eligibility_decision_refused":
        return _refuse_eligibility_decision(
            analysis,
            verification,
            accuracy_warnings,
        )

    if intent == "study_overview":
        return _study_overview_answer(
            analysis,
            verification,
            accuracy_warnings,
        )

    category_map = {
        "eligibility_inclusion": (
            "inclusion_criteria",
            "who may join requirements",
            (
                "Based on the verified study summary, people may be able to "
                "join if these listed requirements apply. Exact wording is "
                "preserved:"
            ),
        ),
        "eligibility_exclusion": (
            "exclusion_criteria",
            "reasons someone may not join",
            (
                "Based on the verified study summary, someone may not be able "
                "to join if these listed reasons apply. Exact wording is "
                "preserved:"
            ),
        ),
        "visits": (
            "visits",
            "visit schedule details",
            (
                "Here is the visit schedule from the verified study summary. "
                "Timing values are shown exactly as extracted:"
            ),
        ),
        "procedures": (
            "procedures",
            "study procedures",
            (
                "Here are study procedures from the verified summary. Names "
                "are shown exactly as extracted:"
            ),
        ),
        "safety": (
            "safety_information",
            "safety information",
            (
                "Here is safety-related information from the verified study "
                "summary. Exact wording is preserved:"
            ),
        ),
    }

    if intent not in category_map:
        return PatientAnswer(
            answer_text=(
                "I can explain general information from the study summary "
                "about who may join, visits, procedures, and safety statements. "
                "I cannot invent clinical facts, give medical advice, or decide "
                "if you personally qualify."
            ),
            intent="unsupported",
            citations=[],
            confidence="unavailable",
            limitations=[
                "Query did not match a supported patient explanation intent.",
            ],
            disclaimers=list(_DEFAULT_DISCLAIMERS),
            audience="patient",
        )

    category, topic_label, preface = category_map[intent]
    citations = _collect_category_citations(analysis, verification, category)
    if intent == "visits" and is_visit_count_query(query):
        preface = build_visit_count_preface(
            analysis,
            verification,
            audience="patient",
        )
        if not citations:
            return PatientAnswer(
                answer_text=preface,
                intent=intent,
                citations=[],
                confidence="unavailable",
                limitations=[
                    "No visits were present in the extracted study summary.",
                ],
                disclaimers=list(_DEFAULT_DISCLAIMERS),
                audience="patient",
            )
    return _build_lookup_answer(
        intent=intent,
        topic_label=topic_label,
        preface=preface,
        citations=citations,
        accuracy_warnings=accuracy_warnings,
    )
