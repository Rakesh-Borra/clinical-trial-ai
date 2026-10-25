"""Version comparison: additions, removals, modifications, evidence and publication gate."""

from __future__ import annotations

import pytest

from services import review_store, version_compare as vc, version_store
from tests.version_helpers import make_guide

V1 = {
    "visits": [
        ("Visit at week 2.", 4, "Follow-up 1 occurs at Week 2 ±3 days."),
        ("Final visit at week 8.", 4, "Final visit occurs at Week 8 ±5 days."),
    ],
    "procedures": [("Blood is drawn at each visit.", 5, "Blood collection occurs at five visits.")],
    "exclusion": [("You cannot join if pregnant.", 3, "Pregnant or breastfeeding women are excluded.")],
}
V2 = {
    "visits": [
        ("Visit at week 2.", 4, "Follow-up 1 occurs at Week 2 ±5 days."),  # modified
        ("Final visit at week 8.", 4, "Final visit occurs at Week 8 ±5 days."),  # unchanged
    ],
    "procedures": [
        ("Blood is drawn at each visit.", 5, "Blood collection occurs at five visits."),
        ("An ECG is done at week 4.", 6, "An electrocardiogram is performed at Week 4."),  # added
    ],
    # exclusion removed
}


def _page_texts(guide_content):
    pages = {}
    for specs in guide_content.values():
        for _, page, excerpt in specs:
            pages[page] = pages.get(page, "") + " " + excerpt.casefold()
    return pages


def _save(pid, content, *, study_id=None, label=None):
    review_store.save_draft(
        pid, make_guide(content), name="Anemia Study", uploaded_name="p.pdf",
        upload_fingerprint="p.pdf:1:abc", study_id=study_id, version_label=label,
    )
    return version_store.get_version(pid)["study_id"]


@pytest.fixture
def two_versions():
    study = _save("v1", V1)
    _save("v2", V2, study_id=study, label="2.0")
    return study


def _loader(old=V1, new=V2):
    texts = {"v1": _page_texts(old), "v2": _page_texts(new)}
    return lambda pid: texts.get(pid)


def test_diff_reports_additions_removals_and_modifications(two_versions):
    diff = vc.compare_versions("v2", page_text_loader=_loader())
    by_kind = {}
    for change in diff["changes"]:
        by_kind.setdefault(change["change"], []).append(change)
    assert len(by_kind["modified"]) == 1 and len(by_kind["added"]) == 1 and len(by_kind["removed"]) == 1
    modified = by_kind["modified"][0]
    assert "±3" in modified["old"]["original_text"] and "±5" in modified["new"]["original_text"]
    assert "ECG" in by_kind["added"][0]["new"]["plain_text"]
    assert by_kind["removed"][0]["section"] == "Who cannot join"
    # explanations reuse the statement wording the agents produced
    assert "Blood" not in " ".join(c["explanation"] for c in diff["changes"])
    assert "An ECG is done at week 4." in by_kind["added"][0]["explanation"]
    assert diff["previous_label"] == "1.0" and diff["new_label"] == "2.0"


def test_identical_versions_have_no_changes():
    study = _save("v1", V1)
    _save("v2", V1, study_id=study, label="2.0")
    assert vc.compare_versions("v2", page_text_loader=_loader(V1, V1))["changes"] == []


def test_changes_are_verified_against_both_pdfs(two_versions):
    diff = vc.compare_versions("v2", page_text_loader=_loader())
    assert diff["source_backed"] is True and diff["notice"] == ""
    assert {c["evidence"] for c in diff["changes"]} == {"verified"}
    assert version_store.load_comparison("v2")["source_verified"] == 1


def test_change_not_on_cited_page_is_unverified_and_blocks_approval(two_versions):
    texts = {"v1": _page_texts(V1), "v2": _page_texts(V2)}
    texts["v2"][6] = "unrelated text"  # the ECG excerpt is not on page 6
    diff = vc.compare_versions("v2", page_text_loader=lambda pid: texts[pid])
    evidence = {c["change"]: c["evidence"] for c in diff["changes"]}
    assert evidence["added"] == "unverified" and evidence["modified"] == "verified"
    assert vc.approval_block_reason(version_store.load_comparison("v2"))
    assert version_store.load_comparison("v2")["source_verified"] == 0


@pytest.mark.parametrize("missing", ["v1", "v2", "both"])
def test_unavailable_pdf_is_never_presented_as_verified(two_versions, missing):
    texts = {"v1": _page_texts(V1), "v2": _page_texts(V2)}
    for pid in ("v1", "v2"):
        if missing in (pid, "both"):
            texts[pid] = None
    diff = vc.compare_versions("v2", page_text_loader=lambda pid: texts[pid])
    assert diff["source_backed"] is False
    assert "unavailable" in diff["notice"] and "NOT been verified" in diff["notice"]
    assert {c["evidence"] for c in diff["changes"]} == {"unavailable"}
    assert version_store.load_comparison("v2")["source_verified"] == 0


def test_default_loader_is_none_when_original_pdf_not_stored(two_versions):
    assert vc.default_page_texts("v1") is None
    diff = vc.compare_versions("v2")  # real loader, no stored PDFs
    assert diff["source_backed"] is False


def test_first_version_cannot_be_compared_and_has_no_gate():
    _save("v1", V1)
    with pytest.raises(version_store.VersionError):
        vc.compare_versions("v1", page_text_loader=_loader())
    assert vc.comparison_block_reason("v1", make_guide(V1)) is None


def test_publication_gate_requires_fresh_approved_comparison(two_versions):
    guide = make_guide(V2)
    assert "Compare this version with Version 1.0" in vc.comparison_block_reason("v2", guide)
    vc.compare_versions("v2", page_text_loader=_loader())
    assert "Approve the changes" in vc.comparison_block_reason("v2", guide)
    version_store.approve_comparison("v2")
    assert vc.comparison_block_reason("v2", guide) is None
    # editing a statement afterwards makes the approved comparison stale
    edited = make_guide(V2)
    edited.procedures.items[0].plain_text = "Blood is drawn five times."
    assert "changed after the comparison" in vc.comparison_block_reason("v2", edited)
    # recomparing clears the approval
    vc.compare_versions("v2", page_text_loader=_loader())
    assert version_store.load_comparison("v2")["approved_at"] is None


def test_rejected_statements_are_ignored(two_versions):
    guide = make_guide(V2)
    guide.procedures.items[1].rejected = True
    review_store.save_draft("v2", guide, name="Anemia Study")
    diff = vc.compare_versions("v2", page_text_loader=_loader())
    assert not [c for c in diff["changes"] if c["change"] == "added"]


def test_v3_compares_with_v2_not_v1(two_versions):
    _save("v3", V2, study_id=two_versions, label="3.0")
    assert version_store.previous_version("v3")["protocol_id"] == "v2"
    diff = vc.compare_versions("v3", page_text_loader=lambda pid: _page_texts(V2))
    assert diff["previous_label"] == "2.0" and diff["changes"] == []


def test_comparison_is_per_version_and_keeps_history(two_versions):
    vc.compare_versions("v2", page_text_loader=_loader())
    assert version_store.load_comparison("v1") is None
    assert review_store.load_draft("v1")["guide"].visits.items[0].original_text.endswith("±3 days.")


def _amendment_pair(v1_content, v2_content, v2_pages):
    """Save two versions; V2 PDF text is supplied explicitly (may omit Timing:)."""
    study = _save("v1", v1_content)
    _save("v2", v2_content, study_id=study, label="2.0")
    texts = {"v1": _page_texts(v1_content), "v2": v2_pages}
    return lambda pid: texts[pid]


def test_day60_followup_with_timing_label_and_paraphrase_passes():
    """Extractor inserts 'Timing:'; PDF says follow-up; plain says check-up."""
    v1 = {
        "visits": [
            ("A screening visit on Day 0.", 2, "Screening Visit. Timing: Day 0"),
        ],
    }
    v2 = {
        "visits": [
            ("A screening visit on Day 0.", 2, "Screening Visit. Timing: Day 0"),
            (
                "There will be an extra check-up visit on Day 60.",
                2,
                "Additional Follow-up Visit. Timing: Day 60",
            ),
        ],
    }
    v2_pages = {
        2: (
            "screening visit: day 0. "
            "additional follow-up visit: day 60. (new in version 2.0)"
        ),
    }
    diff = vc.compare_versions("v2", page_text_loader=_amendment_pair(v1, v2, v2_pages))
    added = next(c for c in diff["changes"] if c["change"] == "added")
    assert "Day 60" in added["new"]["plain_text"]
    assert added["evidence"] == "verified"
    assert diff["source_backed"] is True


def test_incorrect_day90_followup_fails():
    v1 = {"visits": [("A screening visit on Day 0.", 2, "Screening Visit. Timing: Day 0")]}
    v2 = {
        "visits": [
            ("A screening visit on Day 0.", 2, "Screening Visit. Timing: Day 0"),
            (
                "There will be an extra check-up visit on Day 90.",
                2,
                "Additional Follow-up Visit. Timing: Day 90",
            ),
        ],
    }
    v2_pages = {2: "screening visit: day 0. additional follow-up visit: day 60."}
    diff = vc.compare_versions("v2", page_text_loader=_amendment_pair(v1, v2, v2_pages))
    added = next(c for c in diff["changes"] if c["change"] == "added")
    assert added["evidence"] == "unverified"


def test_invented_blood_test_fails():
    v1 = {"procedures": [("Blood is drawn at each visit.", 5, "Blood collection occurs.")]}
    v2 = {
        "procedures": [
            ("Blood is drawn at each visit.", 5, "Blood collection occurs."),
            (
                "A new genetic screening blood test is required.",
                5,
                "Genetic Screening Assay: mandatory DNA panel.",
            ),
        ],
    }
    v2_pages = {5: "blood collection occurs at each visit. hba1c is measured."}
    diff = vc.compare_versions("v2", page_text_loader=_amendment_pair(v1, v2, v2_pages))
    added = next(c for c in diff["changes"] if c["change"] == "added")
    assert added["evidence"] == "unverified"


def test_incorrect_cited_page_for_followup_fails():
    v1 = {"visits": [("A screening visit on Day 0.", 2, "Screening Visit. Timing: Day 0")]}
    v2 = {
        "visits": [
            ("A screening visit on Day 0.", 2, "Screening Visit. Timing: Day 0"),
            (
                "There will be an extra check-up visit on Day 60.",
                3,  # wrong page — Day 60 is only on page 2
                "Additional Follow-up Visit. Timing: Day 60",
            ),
        ],
    }
    v2_pages = {
        2: "screening visit: day 0. additional follow-up visit: day 60.",
        3: "eligibility age 18 to 70 years. no visit schedule on this page.",
    }
    diff = vc.compare_versions("v2", page_text_loader=_amendment_pair(v1, v2, v2_pages))
    added = next(c for c in diff["changes"] if c["change"] == "added")
    assert added["evidence"] == "unverified"


def test_age_changed_18_65_to_18_70_passes():
    v1 = {
        "inclusion": [
            ("You must be 18 to 65 years old.", 1, "Age 18–65 years inclusive."),
        ],
    }
    v2 = {
        "inclusion": [
            ("You must be 18 to 70 years old.", 1, "Age 18–70 years inclusive."),
        ],
    }
    v2_pages = {1: "inclusion criteria. age 18–70 years inclusive."}
    # V1 page must support the old excerpt for the modified side.
    texts = {
        "v1": {1: "inclusion criteria. age 18–65 years inclusive."},
        "v2": v2_pages,
    }
    study = _save("v1", v1)
    _save("v2", v2, study_id=study, label="2.0")
    diff = vc.compare_versions("v2", page_text_loader=lambda pid: texts[pid])
    modified = next(c for c in diff["changes"] if c["change"] == "modified")
    assert "70" in modified["new"]["original_text"]
    assert modified["evidence"] == "verified"


def test_hba1c_addition_passes():
    v1 = {"procedures": [("Blood is drawn at each visit.", 5, "Blood collection occurs.")]}
    v2 = {
        "procedures": [
            ("Blood is drawn at each visit.", 5, "Blood collection occurs."),
            (
                "An HbA1c blood test is added.",
                5,
                "HbA1c blood test at screening.",
            ),
        ],
    }
    v2_pages = {5: "blood collection occurs. hba1c blood test at screening."}
    diff = vc.compare_versions("v2", page_text_loader=_amendment_pair(v1, v2, v2_pages))
    added = next(c for c in diff["changes"] if c["change"] == "added")
    assert "HbA1c" in added["new"]["plain_text"] or "hba1c" in added["new"]["original_text"].casefold()
    assert added["evidence"] == "verified"
