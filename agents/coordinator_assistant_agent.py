
"""
Deterministic Coordinator Assistant.

Retrieves verified protocol facts for coordinators. Does not call an LLM,
does not invent clinical content, and does not decide patient eligibility.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from models.coordinator_models import CoordinatorAnswer, CoordinatorCitation
from models.protocol_models import ProtocolAnalysis
from services.source_verifier import normalize_text

_DEFAULT_DISCLAIMERS = [
    (
        "This assistant retrieves protocol information for coordinator "
        "reference only. It does not provide medical advice."
    ),
    (
        "It does not determine final patient eligibility. Eligibility "
        "decisions require qualified clinical judgment and the full protocol."
    ),
]

_VERIFIED_STATUS = "exact_match"
_UNVERIFIED_STATUSES = {
    "missing_page",
    "invalid_page",
    "not_verified",
}

_ELIGIBILITY_DECISION_PATTERNS = (
    r"\bis (this |the )?patient eligible\b",
    r"\bcan (they|the patient|this patient) enroll\b",
    r"\bshould (we|i) enroll\b",
    r"\bdetermine eligibility\b",
    r"\beligibility decision\b",
    r"\bam i eligible\b",
)

_MEDICAL_ADVICE_PATTERNS = (
    r"\bdiagnose\b",
    r"\btreat(?:ment)? advice\b",
    r"\bmedical advice\b",
)

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "eligibility_inclusion": (
        "inclusion",
        "include",
        "eligible criteria",
        "eligibility criteria",
        "who can join",
        "who may enroll",
    ),
    "eligibility_exclusion": (
        "exclusion",
        "exclude",
        "who cannot",
        "who can't",
        "not allowed",
    ),
    "visits": (
        "visit",
        "schedule",
        "day 0",
        "day 30",
        "timing",
        "follow-up",
        "screening visit",
    ),
    "procedures": (
        "procedure",
        "procedures",
        "testing",
        "assessment",
        "blood glucose",
        "blood pressure",
    ),
    "safety": (
        "safety",
        "adverse",
        "risk",
        "hypoglycemia",
        "monitor",
        "side effect",
    ),
    "metadata": (
        "title",
        "protocol id",
        "protocol name",
        "study name",
        "trial name",
        "what study",
    ),
}


def _matches_any(query: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_text(query)
    return any(re.search(pattern, normalized) for pattern in patterns)


def classify_coordinator_intent(query: str) -> str:
    """Classify coordinator intent with deterministic keyword rules."""
    if _matches_any(query, _MEDICAL_ADVICE_PATTERNS):
        return "medical_advice_refused"
    if _matches_any(query, _ELIGIBILITY_DECISION_PATTERNS):
        return "eligibility_decision_refused"

    normalized = normalize_text(query)

    # Prefer exclusion before inclusion when both words appear.
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


def _citation(
    category: str,
    statement: str,
    source_page: Optional[int],
    status: str,
) -> CoordinatorCitation:
    return CoordinatorCitation(
        category=category,
        statement=statement,
        source_page=source_page,
        verification_status=status,
    )


def _collect_category_citations(
    analysis: ProtocolAnalysis,
    verification: Sequence[dict],
    category: str,
) -> List[CoordinatorCitation]:
    index = _verification_index(verification)
    citations: list[CoordinatorCitation] = []

    if category == "inclusion_criteria":
        for item in analysis.inclusion_criteria:
            status = _status_for(index, category, item.description)
            citations.append(
                _citation(category, item.description, item.source_page, status)
            )
    elif category == "exclusion_criteria":
        for item in analysis.exclusion_criteria:
            status = _status_for(index, category, item.description)
            citations.append(
                _citation(category, item.description, item.source_page, status)
            )
    elif category == "procedures":
        for item in analysis.procedures:
            status = _status_for(index, category, item.name)
            statement = item.name
            if item.description:
                statement = f"{item.name} — {item.description}"
            citations.append(
                _citation(category, statement, item.source_page, status)
            )
    elif category == "visits":
        for item in analysis.visits:
            status = _status_for(index, category, item.visit_name)
            timing = item.timing or "timing not stated"
            procs = (
                f"; procedures: {', '.join(item.procedures)}"
                if item.procedures
                else ""
            )
            statement = f"{item.visit_name}: {timing}{procs}"
            citations.append(
                _citation(category, statement, item.source_page, status)
            )
    elif category == "safety_information":
        for item in analysis.safety_information:
            status = _status_for(index, category, item.description)
            citations.append(
                _citation(category, item.description, item.source_page, status)
            )
    elif category == "metadata":
        # Metadata is not covered by source_verifier list checks; expose as
        # analysis-backed values without inventing page numbers.
        if analysis.trial_title:
            citations.append(
                _citation(
                    "metadata",
                    f"Trial title: {analysis.trial_title}",
                    None,
                    "from_analysis",
                )
            )
        if analysis.protocol_id:
            citations.append(
                _citation(
                    "metadata",
                    f"Protocol ID: {analysis.protocol_id}",
                    None,
                    "from_analysis",
                )
            )

    return citations


def _is_preferred_status(status: str) -> bool:
    return status in {_VERIFIED_STATUS, "from_analysis"}


def _split_verified(
    citations: Sequence[CoordinatorCitation],
) -> tuple[list[CoordinatorCitation], list[CoordinatorCitation]]:
    verified = [c for c in citations if _is_preferred_status(c.verification_status)]
    unverified = [
        c for c in citations if not _is_preferred_status(c.verification_status)
    ]
    return verified, unverified


def _format_citation_lines(citations: Iterable[CoordinatorCitation]) -> list[str]:
    lines: list[str] = []
    for citation in citations:
        page = (
            f"page {citation.source_page}"
            if citation.source_page is not None
            else "page not assigned"
        )
        lines.append(
            f"- {citation.statement} "
            f"({page}; verification: {citation.verification_status})"
        )
    return lines


def _relevant_accuracy_warnings(
    accuracy_warnings: Optional[Sequence[str]],
    intent: str,
) -> list[str]:
    if not accuracy_warnings:
        return []

    relevant: list[str] = []
    for warning in accuracy_warnings:
        normalized = normalize_text(warning)
        if intent in {"eligibility_inclusion", "eligibility_exclusion"}:
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
        elif "trial_title" in normalized or "protocol_id" in normalized:
            if intent == "metadata":
                relevant.append(warning)
    return relevant


def _build_lookup_answer(
    *,
    intent: str,
    category_label: str,
    citations: List[CoordinatorCitation],
    accuracy_warnings: Optional[Sequence[str]],
) -> CoordinatorAnswer:
    verified, unverified = _split_verified(citations)
    limitations: list[str] = []
    limitations.extend(
        _relevant_accuracy_warnings(accuracy_warnings, intent)
    )

    if not citations:
        return CoordinatorAnswer(
            answer_text=(
                f"No {category_label} were found in the extracted protocol "
                "analysis. No clinical facts were invented."
            ),
            intent=intent,
            citations=[],
            confidence="unavailable",
            limitations=limitations
            or [
                f"Extracted analysis contained no {category_label}.",
            ],
            disclaimers=list(_DEFAULT_DISCLAIMERS),
        )

    preferred = verified if verified else unverified
    lines = _format_citation_lines(preferred)

    if verified and not unverified:
        confidence = "high"
        preface = f"Verified {category_label} from the protocol analysis:"
    elif verified and unverified:
        confidence = "partial"
        preface = (
            f"Verified {category_label} from the protocol analysis "
            "(additional unverified items are listed in limitations):"
        )
        limitations.append(
            f"{len(unverified)} {category_label} item(s) were not "
            f"{_VERIFIED_STATUS} and were not treated as confirmed facts."
        )
        for item in unverified:
            limitations.append(
                f"Unverified ({item.verification_status}): {item.statement}"
            )
    else:
        confidence = "partial"
        preface = (
            f"{category_label.capitalize()} were found but none had "
            f"{_VERIFIED_STATUS} verification. Treat as unverified:"
        )
        limitations.append(
            "No exact_match verification was available for these statements."
        )

    # Prefer returning verified citations in the answer payload; include
    # unverified only when no verified facts exist.
    answer_citations = list(verified) if verified else list(unverified)

    return CoordinatorAnswer(
        answer_text=preface + "\n" + "\n".join(lines),
        intent=intent,
        citations=answer_citations,
        confidence=confidence,
        limitations=limitations,
        disclaimers=list(_DEFAULT_DISCLAIMERS),
    )


def _refuse_medical_advice() -> CoordinatorAnswer:
    return CoordinatorAnswer(
        answer_text=(
            "I cannot provide medical advice, diagnosis, or treatment "
            "recommendations. I can only retrieve protocol facts such as "
            "eligibility criteria, visits, procedures, and safety statements "
            "for coordinator review."
        ),
        intent="medical_advice_refused",
        citations=[],
        confidence="unavailable",
        limitations=[
            "Medical advice and clinical decision-making are out of scope.",
        ],
        disclaimers=list(_DEFAULT_DISCLAIMERS),
    )


def _refuse_eligibility_decision(
    analysis: ProtocolAnalysis,
    verification: Sequence[dict],
    accuracy_warnings: Optional[Sequence[str]],
) -> CoordinatorAnswer:
    inclusions = _collect_category_citations(
        analysis, verification, "inclusion_criteria"
    )
    exclusions = _collect_category_citations(
        analysis, verification, "exclusion_criteria"
    )
    verified_inclusions, _ = _split_verified(inclusions)
    verified_exclusions, _ = _split_verified(exclusions)

    lines = [
        "I cannot determine whether a specific patient is eligible or "
        "provide a final enrollment decision.",
        "Below are protocol criteria for coordinator review only:",
    ]
    if verified_inclusions:
        lines.append("Verified inclusion criteria:")
        lines.extend(_format_citation_lines(verified_inclusions))
    if verified_exclusions:
        lines.append("Verified exclusion criteria:")
        lines.extend(_format_citation_lines(verified_exclusions))
    if not verified_inclusions and not verified_exclusions:
        lines.append(
            "No verified inclusion/exclusion criteria were available "
            "in the extracted analysis."
        )

    citations = list(verified_inclusions) + list(verified_exclusions)
    limitations = [
        "Final eligibility determination is out of scope for this assistant.",
    ]
    limitations.extend(
        _relevant_accuracy_warnings(
            accuracy_warnings,
            "eligibility_inclusion",
        )
    )

    confidence = "high" if citations else "unavailable"
    return CoordinatorAnswer(
        answer_text="\n".join(lines),
        intent="eligibility_decision_refused",
        citations=citations,
        confidence=confidence,
        limitations=limitations,
        disclaimers=list(_DEFAULT_DISCLAIMERS),
    )


def answer_coordinator_query(
    query: str,
    analysis: ProtocolAnalysis,
    verification: list[dict],
    *,
    accuracy_warnings: Optional[list[str]] = None,
) -> CoordinatorAnswer:
    """
    Answer a coordinator question using extracted analysis and verification.

    Deterministic only: no LLM calls and no invented clinical content.
    """
    if not query or not query.strip():
        return CoordinatorAnswer(
            answer_text="Please provide a coordinator question about the protocol.",
            intent="unsupported",
            citations=[],
            confidence="unavailable",
            limitations=["Empty query."],
            disclaimers=list(_DEFAULT_DISCLAIMERS),
        )

    intent = classify_coordinator_intent(query)

    if intent == "medical_advice_refused":
        return _refuse_medical_advice()

    if intent == "eligibility_decision_refused":
        return _refuse_eligibility_decision(
            analysis,
            verification,
            accuracy_warnings,
        )

    category_map = {
        "eligibility_inclusion": ("inclusion_criteria", "inclusion criteria"),
        "eligibility_exclusion": ("exclusion_criteria", "exclusion criteria"),
        "visits": ("visits", "visit schedule details"),
        "procedures": ("procedures", "procedures"),
        "safety": ("safety_information", "safety statements"),
        "metadata": ("metadata", "trial metadata"),
    }

    if intent not in category_map:
        return CoordinatorAnswer(
            answer_text=(
                "I can help look up inclusion criteria, exclusion criteria, "
                "visits, procedures, safety information, and trial metadata "
                "from the verified protocol analysis. I cannot invent "
                "clinical facts or make eligibility decisions."
            ),
            intent="unsupported",
            citations=[],
            confidence="unavailable",
            limitations=[
                "Query did not match a supported coordinator lookup intent.",
            ],
            disclaimers=list(_DEFAULT_DISCLAIMERS),
        )

    category, label = category_map[intent]
    citations = _collect_category_citations(analysis, verification, category)
    return _build_lookup_answer(
        intent=intent,
        category_label=label,
        citations=citations,
        accuracy_warnings=accuracy_warnings,
    )
