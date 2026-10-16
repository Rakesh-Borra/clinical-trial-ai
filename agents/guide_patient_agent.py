"""
Patient Assistant that answers only from the approved Patient Guide.

Patients must never see protocol content that a coordinator has not
approved. This assistant therefore answers from the approved items of the
Plain-Language Patient Guide, never from the full protocol analysis. If a
topic has not been approved yet, it says the study team is still reviewing
it.

Deterministic: no LLM calls. It reuses the Patient Assistant's intent
rules, so eligibility-decision and medical-advice questions are refused
exactly as before.
"""

from __future__ import annotations

from typing import List, Optional

from agents.patient_agent import classify_patient_intent
from models.patient_models import PatientAnswer, PatientCitation
from models.plain_language_models import GuideItem, GuideSection, PatientGuide

APPROVED_STATUS = "coordinator_approved"

_DISCLAIMERS = [
    (
        "This assistant only shares information the study team has approved "
        "for patients. It is general information, not medical advice."
    ),
    (
        "It cannot decide whether you are eligible for a study. Talk with "
        "the study team or a qualified clinician about your personal situation."
    ),
]

_ASK_ABOUT = (
    "You can ask me about who may be able to join, who may not, the study "
    "visits, the tests and procedures, or safety."
)

# intent -> (guide section attribute, topic in everyday words, friendly opener)
_SECTION_INTENTS = {
    "eligibility_inclusion": (
        "who_can_join",
        "who may be able to join",
        "Here is who may be able to join, as approved by the study team:",
    ),
    "eligibility_exclusion": (
        "who_cannot_join",
        "who may not be able to join",
        "Here are reasons someone may not be able to take part, as approved "
        "by the study team:",
    ),
    "visits": (
        "visits",
        "the study visits",
        "Here is what the study team has shared about the visits:",
    ),
    "procedures": (
        "procedures",
        "the tests and procedures",
        "Here are the tests and procedures the study team has described:",
    ),
    "safety": (
        "safety",
        "safety",
        "Here is what the study team has shared about safety:",
    ),
}


def _answer(
    text: str,
    intent: str,
    *,
    citations: Optional[List[PatientCitation]] = None,
    confidence: str = "unavailable",
    limitations: Optional[List[str]] = None,
) -> PatientAnswer:
    return PatientAnswer(
        answer_text=text,
        intent=intent,
        citations=citations or [],
        confidence=confidence,
        limitations=limitations or [],
        disclaimers=list(_DISCLAIMERS),
        audience="patient",
    )


def _approved(section: GuideSection) -> List[GuideItem]:
    return [item for item in section.items if item.approved]


def _line(item: GuideItem) -> str:
    page = f" (protocol page {item.source_page})" if item.source_page else ""
    return f"- {item.plain_text}{page}"


def _citation(item: GuideItem) -> PatientCitation:
    return PatientCitation(
        category=item.section,
        statement=item.plain_text,
        source_page=item.source_page,
        verification_status=APPROVED_STATUS,
        plain_language=item.plain_text,
    )


def _pending_note(count: int) -> str:
    if count == 1:
        return "One more item on this topic is still being reviewed by the study team, so it is not shown yet."
    return f"{count} more items on this topic are still being reviewed by the study team, so they are not shown yet."


def _under_review(topic: str) -> str:
    return (
        f"The study team is still reviewing the information about {topic}, "
        "so I can't share it yet. Please check back later or ask the study "
        "team directly."
    )


def _not_published() -> PatientAnswer:
    return _answer(
        "The study team hasn't approved a patient guide for this study yet, "
        "so I can't answer questions about it yet. Please check back later "
        "or ask the study team directly.",
        "not_yet_approved",
        limitations=["No approved patient guide content is available yet."],
    )


def _section_answer(guide: PatientGuide, intent: str) -> PatientAnswer:
    attr, topic, opener = _SECTION_INTENTS[intent]
    section: GuideSection = getattr(guide, attr)
    approved = _approved(section)
    pending = len(section.items) - len(approved)

    if not section.items:
        return _answer(
            f"The patient guide doesn't include details about {topic}. "
            "Please ask the study team about this.",
            intent,
            limitations=[f"The guide has no items about {topic}."],
        )
    if not approved:
        return _answer(
            _under_review(topic),
            intent,
            limitations=[f"{pending} item(s) about {topic} are awaiting approval."],
        )

    lines = [opener, ""] + [_line(item) for item in approved]
    limitations: List[str] = []
    if pending:
        lines += ["", _pending_note(pending)]
        limitations.append(f"{pending} item(s) about {topic} are awaiting approval.")
    if intent == "eligibility_inclusion":
        lines += ["", "Meeting these does not decide anything on its own. Only the study team can confirm who can join."]
    return _answer(
        "\n".join(lines),
        intent,
        citations=[_citation(item) for item in approved],
        confidence="high" if not pending else "partial",
        limitations=limitations,
    )


def _eligibility_refusal(guide: PatientGuide) -> PatientAnswer:
    inclusions = _approved(guide.who_can_join)
    exclusions = _approved(guide.who_cannot_join)
    pending = sum(
        len(s.items) - len(_approved(s))
        for s in (guide.who_can_join, guide.who_cannot_join)
    )

    lines = [
        "I can't tell you whether you personally are eligible for this "
        "study. That depends on your individual situation, and only the "
        "study team can decide.",
    ]
    if inclusions or exclusions:
        lines += ["", "Here are the general criteria the study team has approved, which you can talk through with them:"]
    if inclusions:
        lines += ["", "**Who may be able to join:**"] + [_line(i) for i in inclusions]
    if exclusions:
        lines += ["", "**Who may not be able to join:**"] + [_line(i) for i in exclusions]
    if not inclusions and not exclusions:
        lines += ["", "The study team is still reviewing the joining criteria, so I can't share them yet."]
    elif pending:
        lines += ["", _pending_note(pending)]

    return _answer(
        "\n".join(lines),
        "eligibility_decision_refused",
        citations=[_citation(i) for i in inclusions + exclusions],
        confidence="high" if (inclusions or exclusions) else "unavailable",
        limitations=["This assistant does not evaluate personal eligibility."],
    )


def _medical_advice_refusal() -> PatientAnswer:
    return _answer(
        "I can't give medical advice, a diagnosis, or treatment "
        "recommendations. I can only share general information the study "
        "team has approved, such as visits, procedures, and the listed "
        "criteria. Please talk with the study team or a qualified clinician "
        "about your health.",
        "medical_advice_refused",
        limitations=["Medical advice and personal clinical decisions are out of scope."],
    )


def _overview(guide: PatientGuide) -> PatientAnswer:
    title = guide.trial_title or "this clinical trial"
    protocol = f" (protocol ID {guide.protocol_id})" if guide.protocol_id else ""
    return _answer(
        f"This guide is for the study “{title}”{protocol}. {_ASK_ABOUT}",
        "study_overview",
        confidence="high",
    )


def answer_patient_from_guide(query: str, guide: Optional[PatientGuide]) -> PatientAnswer:
    """Answer a patient question using approved Patient Guide items only."""
    if not query or not query.strip():
        return _answer(f"Please ask a question about the study. {_ASK_ABOUT}", "unsupported",
                       limitations=["Empty query."])

    intent = classify_patient_intent(query)

    # Refusals never depend on what has been approved.
    if intent == "medical_advice_refused":
        return _medical_advice_refusal()

    if guide is None or guide.approved_count() == 0:
        return _not_published()

    if intent == "eligibility_decision_refused":
        return _eligibility_refusal(guide)
    if intent == "study_overview":
        return _overview(guide)
    if intent in _SECTION_INTENTS:
        return _section_answer(guide, intent)

    return _answer(
        "I'm not sure I understood that question. " + _ASK_ABOUT + " I can't "
        "give medical advice or decide whether you personally can join.",
        "unsupported",
        limitations=["Query did not match a supported patient question."],
    )
