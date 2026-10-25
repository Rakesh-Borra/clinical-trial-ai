"""Offline tests for approved-only BM25 patient-guide retrieval."""

from __future__ import annotations

from models.plain_language_models import GuideItem, GuideSection, PatientGuide
from services.guide_retrieval import (
    approved_items,
    retrieve_approved_passages,
    tokenize,
)


def _item(
    plain: str,
    *,
    section: str = "procedures",
    approved: bool = True,
    rejected: bool = False,
    page: int = 3,
    item_id: str = "",
    original: str | None = None,
) -> GuideItem:
    return GuideItem(
        section=section,  # type: ignore[arg-type]
        original_text=original or plain,
        plain_text=plain,
        source_page=page,
        approved=approved,
        rejected=rejected,
        item_id=item_id or plain[:24],
        source_status="verified",
        analysis_complete=True,
    )


def _section(title: str, items: list[GuideItem]) -> GuideSection:
    return GuideSection(title=title, introduction="", items=items)


def _guide(*items: GuideItem) -> PatientGuide:
    by_section: dict[str, list[GuideItem]] = {
        "inclusion_criteria": [],
        "exclusion_criteria": [],
        "visits": [],
        "procedures": [],
        "safety_information": [],
    }
    for item in items:
        by_section.setdefault(item.section, []).append(item)
    return PatientGuide(
        who_can_join=_section("Who may be able to join", by_section["inclusion_criteria"]),
        who_cannot_join=_section(
            "Who may not be able to join", by_section["exclusion_criteria"]
        ),
        visits=_section("Trial visits", by_section["visits"]),
        procedures=_section("Tests and procedures", by_section["procedures"]),
        safety=_section("Safety information", by_section["safety_information"]),
        disclaimer="General study information only.",
        analysis_complete=True,
    )


def test_tokenize_drops_stopwords():
    assert "glucose" in tokenize("What about the blood glucose test?")
    assert "the" not in tokenize("What about the blood glucose test?")


def test_approved_items_exclude_pending_and_rejected():
    guide = _guide(
        _item("You may have a blood glucose test.", approved=True, item_id="ok"),
        _item("Pending MRI scan details.", approved=False, item_id="pending"),
        _item("Rejected dose change note.", approved=True, rejected=True, item_id="rej"),
    )
    ids = {item.item_id for item in approved_items(guide)}
    assert ids == {"ok"}


def test_retrieval_ranks_relevant_approved_statement_first():
    guide = _guide(
        _item(
            "Adults 18 to 75 years old may be able to join.",
            section="inclusion_criteria",
            page=2,
            item_id="age",
        ),
        _item(
            "You may have a fasting blood glucose test at the clinic.",
            section="procedures",
            page=8,
            item_id="glucose",
        ),
        _item(
            "There is a follow-up visit on Day 30.",
            section="visits",
            page=5,
            item_id="visit",
        ),
    )
    hits = retrieve_approved_passages(guide, "Do I need a blood glucose test?")
    assert hits
    assert hits[0].item_id == "glucose"
    assert hits[0].source_page == 8
    assert "glucose" in hits[0].plain_text.lower()


def test_retrieval_never_returns_unapproved_content():
    guide = _guide(
        _item(
            "Approved: fasting blood glucose check.",
            approved=True,
            item_id="ok",
        ),
        _item(
            "Unapproved secret glucose protocol dose 999 mg.",
            approved=False,
            item_id="bad",
        ),
    )
    hits = retrieve_approved_passages(guide, "glucose dose protocol")
    assert hits
    assert all(row.item_id == "ok" for row in hits)
    assert all("999" not in row.plain_text for row in hits)


def test_protocol_isolation_uses_only_provided_guide():
    study_a = _guide(
        _item(
            "Study A visits happen every Monday at the hospital.",
            section="visits",
            item_id="a-visit",
        )
    )
    study_b = _guide(
        _item(
            "Study B visits happen every Friday at the clinic.",
            section="visits",
            item_id="b-visit",
        )
    )
    hits_a = retrieve_approved_passages(study_a, "when are the visits")
    hits_b = retrieve_approved_passages(study_b, "when are the visits")
    assert hits_a and hits_a[0].item_id == "a-visit"
    assert hits_b and hits_b[0].item_id == "b-visit"
    assert "Friday" not in hits_a[0].plain_text
    assert "Monday" not in hits_b[0].plain_text


def test_unsupported_query_returns_no_passages():
    guide = _guide(
        _item("There is a follow-up visit on Day 30.", section="visits", item_id="v")
    )
    assert retrieve_approved_passages(guide, "zxqv blorp quantum") == []
