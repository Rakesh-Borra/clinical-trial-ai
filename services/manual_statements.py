"""Coordinator-written protocol statements.

A manual statement uses the normal GuideItem model, so it is stored with the
rest of the protocol version's draft and goes through the same review,
approval and publication rules. It starts Pending and is never marked verified
unless its wording is supported by the text of the cited protocol page.
"""

from __future__ import annotations

import re
import uuid
from typing import Mapping, Optional

from agents.plain_language_agent import reverify_item
from models.plain_language_models import GuideItem, PatientGuide

CATEGORIES = (
    "Eligibility",
    "Study interventions",
    "Study visits and schedule",
    "Procedures and blood collection",
    "Risks and safety",
    "Consent and withdrawal",
    "Other",
)
ELIGIBILITY_CHOICES = {
    "Who can join": "inclusion_criteria",
    "Who cannot join": "exclusion_criteria",
}
_SECTION_BY_CATEGORY = {
    "Study interventions": "study_interventions",
    "Study visits and schedule": "visits",
    "Procedures and blood collection": "procedures",
    "Risks and safety": "safety_information",
    "Consent and withdrawal": "consent_withdrawal",
    "Other": "study_purpose",
}
SECTION_ATTR = {
    "inclusion_criteria": "who_can_join",
    "exclusion_criteria": "who_cannot_join",
    "visits": "visits",
    "procedures": "procedures",
    "safety_information": "safety",
    "study_interventions": "interventions",
    "consent_withdrawal": "consent",
    "study_purpose": "purpose",
}
MAX_LENGTH = 1200


class ManualStatementError(ValueError):
    """The statement cannot be added; the message is shown to the coordinator."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def section_for(category: str, eligibility: str = "Who can join") -> str:
    if category == "Eligibility":
        if eligibility not in ELIGIBILITY_CHOICES:
            raise ManualStatementError("Choose whether the criterion is for who can or cannot join.")
        return ELIGIBILITY_CHOICES[eligibility]
    if category not in _SECTION_BY_CATEGORY:
        raise ManualStatementError("Choose a category.")
    return _SECTION_BY_CATEGORY[category]


def validate(
    guide: PatientGuide,
    section: str,
    text: str,
    page: object,
    source_pages: Optional[Mapping[str, str]],
) -> tuple[str, int]:
    """Return the cleaned text and page, or raise ManualStatementError."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ManualStatementError("The statement cannot be empty.")
    if len(cleaned) > MAX_LENGTH:
        raise ManualStatementError(f"Keep the statement under {MAX_LENGTH} characters.")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ManualStatementError("Enter the protocol page number (1 or higher).")
    if source_pages and str(page) not in source_pages:
        known = sorted(int(p) for p in source_pages if str(p).isdigit())
        span = f" (the protocol has pages {known[0]}–{known[-1]})" if known else ""
        raise ManualStatementError(f"Page {page} is not in this protocol{span}.")
    for item in guide.all_items():
        if (
            getattr(item, "section", None) == section
            and item.source_page == page
            and _norm(item.plain_text) == _norm(cleaned)
        ):
            raise ManualStatementError("This statement was already added for that page.")
    return cleaned, page


def add_manual_statement(
    guide: PatientGuide,
    category: str,
    text: str,
    page: object,
    source_pages: Optional[Mapping[str, str]],
    eligibility: str = "Who can join",
) -> GuideItem:
    """
    Append a Pending statement to the guide and check it against its page.

    A page number is not evidence: the statement counts as verified only when
    the cited page's text supports the wording. Without page text it stays
    unverified and says so; no citation or excerpt is invented.
    """
    section = section_for(category, eligibility)
    cleaned, page = validate(guide, section, text, page, source_pages)
    item = GuideItem(
        section=section,
        original_text=cleaned,
        plain_text=cleaned,
        source_page=page,
        generated_by="coordinator",
        added_by_coordinator=True,
        item_id=uuid.uuid4().hex[:12],
        verification="needs_reverification",
        source_status="unverified",
        source_issue="Added by the coordinator; not yet checked against the protocol.",
    )
    page_text = (source_pages or {}).get(str(page))
    problems = reverify_item(item, page_text)
    if problems:
        item.source_issue = (
            "Source verification is unavailable: the protocol text is not stored for this protocol."
            if not page_text
            else " ".join(problems)
        )
    else:
        item.superseded_text = None
    target = getattr(guide, SECTION_ATTR[section])
    target.items.append(item)
    target.unavailable_message = None
    return item
