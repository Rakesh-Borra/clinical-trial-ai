"""
Review decisions and the patient-facing boundary for the Patient Guide.

Every statement has exactly one review state:

- Pending:  no decision yet (blocks publication)
- Approved: accepted without changes
- Edited:   changed by the coordinator and then explicitly approved
- Rejected: deliberately excluded (never published, never answerable)

These helpers hold no Streamlit state so the rules are easy to test. The
published snapshot is built here, which is what keeps rejected statements out
of both the patient guide and the patient assistant's context.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from agents.plain_language_agent import (
    REJECTED_REASON,
    approval_block_reason,
    active_overrides,
    item_cleared,
    safety_blockers,
)
from models.patient_models import PatientAnswer
from models.plain_language_models import GlossaryTerm, GuideItem, PatientGuide

PENDING = "pending"
APPROVED = "approved"
EDITED = "edited"
REJECTED = "rejected"
ALL = "all"

FILTERS = (
    (ALL, "All Statements"),
    (PENDING, "Pending"),
    (APPROVED, "Approved"),
    (EDITED, "Edited"),
    (REJECTED, "Rejected"),
)

# Sections a patient must not be left without when the protocol described them.
MANDATORY_SECTIONS = (
    ("who_cannot_join", "Who may not be able to join"),
    ("safety", "Safety information"),
)

WITHHELD_MESSAGE = (
    "This information is not available in the approved patient guide. "
    "Please contact the clinical trial coordinator."
)

# Wording that points at other statements; it cannot stand if a neighbour is rejected.
_CROSS_REFERENCE = re.compile(
    r"\b(see above|see below|as (?:described|listed|noted|mentioned) (?:above|below)"
    r"|the (?:above|following) (?:criteria|conditions|exceptions?)"
    r"|except as (?:noted|described)|in addition to the (?:above|following))\b",
    re.IGNORECASE,
)


def review_state(item: GuideItem) -> str:
    """
    The coordinator's decision. AI verification never changes it, but an
    approval whose warnings are not covered by a valid override does not
    count as a decision.
    """
    if item.rejected:
        return REJECTED
    if item.approved and item_cleared(item):
        return EDITED if item.generated_by == "coordinator" else APPROVED
    return PENDING


def state_counts(guide: PatientGuide) -> Dict[str, int]:
    counts = {key: 0 for key, _ in FILTERS}
    counts["edited_total"] = 0
    for item in guide.all_items():
        counts[ALL] += 1
        counts[review_state(item)] += 1
        if item.generated_by == "coordinator":
            counts["edited_total"] += 1
    return counts


def matches_filter(item: GuideItem, key: str) -> bool:
    return key == ALL or review_state(item) == key


def reconsider_item(item: GuideItem) -> None:
    """Return a rejected statement to Pending so it can be decided again."""
    item.rejected = False
    item.approved = False
    item.review_reasons = [r for r in item.review_reasons if r != REJECTED_REASON]
    item.needs_review = bool(item.review_reasons)


def edit_problems(item: GuideItem) -> List[str]:
    """Safety problems of a coordinator-edited statement (never overridable)."""
    return safety_blockers(item)


def override_confirmation_reason(guide: PatientGuide) -> Optional[str]:
    """Human overrides must be confirmed as a set before publishing."""
    overrides = active_overrides(guide)
    ids = sorted(i.item_id for i in overrides)
    if overrides and ids != sorted(guide.overrides_confirmed):
        return (
            f"Cannot publish: {len(overrides)} statement(s) were approved over "
            "an AI verification warning. Review and confirm these human "
            "overrides before publishing."
        )
    return None


# ---------------------------------------------------------------------------
# Publication rules that depend on the decisions
# ---------------------------------------------------------------------------

def decision_block_reason(guide: PatientGuide) -> Optional[str]:
    """Explain why the decisions made so far do not allow publication."""
    items = guide.all_items()
    if not guide.generation_complete:
        return (
            "Cannot publish: statements are still being generated "
            f"({len(items)} of {guide.generation_total or '?'} so far) or some "
            "failed. Wait for generation to finish, or retry the failed ones."
        )
    pending = sum(review_state(i) == PENDING for i in items)
    if pending:
        return (
            f"Cannot publish: {pending} statement(s) still need a coordinator "
            "decision. Approve, edit and approve, or reject each one."
        )

    kept = [i for i in items if not i.rejected]
    if not kept:
        return (
            "Cannot publish: every statement is rejected. Approve at least "
            "one statement."
        )

    for item in kept:
        problems = edit_problems(item)
        if problems:
            return (
                "Cannot publish: an edited statement failed verification "
                f"({problems[0]}) Fix the wording or reject it."
            )

    reason = override_confirmation_reason(guide)
    if reason:
        return reason

    for attr, label in MANDATORY_SECTIONS:
        section = getattr(guide, attr)
        if section.items and all(i.rejected for i in section.items):
            return (
                f"Cannot publish: every statement in “{label}” is rejected, "
                "which would leave a required safety disclosure missing. "
                "Approve at least one statement in that section."
            )

    for section in guide.sections():
        if not any(i.rejected for i in section.items):
            continue
        for item in section.items:
            if not item.rejected and _CROSS_REFERENCE.search(item.plain_text):
                return (
                    f"Cannot publish: a statement in “{section.title}” refers "
                    "to other content (“"
                    f"{_CROSS_REFERENCE.search(item.plain_text).group(0)}”) "
                    "but a related statement in that section was rejected. "
                    "Edit that statement or reject it too."
                )
    return None


def build_published_snapshot(guide: PatientGuide) -> PatientGuide:
    """
    Copy of the guide holding only approved, non-rejected, source-verified
    statements.

    Rejected statements, their original protocol text and their glossary
    terms are absent from the copy, so nothing downstream can retrieve them.
    """
    snapshot = guide.model_copy(deep=True)
    snapshot.generation_complete = guide.generation_complete
    seen: set[str] = set()
    glossary: List[GlossaryTerm] = []
    for section in snapshot.sections():
        kept = [
            i for i in section.items
            if i.approved and not i.rejected and item_cleared(i)
        ]
        if len(kept) != len(section.items):
            section.items = kept
            if not kept:
                section.unavailable_message = WITHHELD_MESSAGE
        for item in kept:
            for term in item.terms_explained:
                if term.term not in seen:
                    seen.add(term.term)
                    glossary.append(term)
    snapshot.glossary = glossary
    return snapshot


def patient_visible_guide(guide: PatientGuide) -> PatientGuide:
    """Defensive filter for anything patient-facing: never expose a rejected item."""
    if any(
        item.rejected or not item.approved or not item_cleared(item)
        for item in guide.all_items()
    ):
        return build_published_snapshot(guide)
    return guide


_NO_REFERRAL_INTENTS = {
    "medical_advice_refused",
    "eligibility_decision_refused",
    "not_yet_approved",
    "study_overview",
    "term_definition",
    "conversational",
    "clarification",
    "unsupported",
}


def with_coordinator_referral(answer: PatientAnswer) -> PatientAnswer:
    """
    Make "no approved information" replies point to the coordinator.

    Applies only to topic questions that found nothing citable; refusals,
    greetings, definitions and the generic fallback are left unchanged.
    """
    if (
        answer.citations
        or answer.intent in _NO_REFERRAL_INTENTS
        or answer.confidence != "unavailable"
        or WITHHELD_MESSAGE in answer.answer_text
    ):
        return answer
    return answer.model_copy(
        update={"answer_text": f"{answer.answer_text}\n\n{WITHHELD_MESSAGE}"}
    )
