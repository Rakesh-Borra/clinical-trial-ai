"""Regression tests for source grounding, per-statement approval, coverage
gating and coordinator summary editing. Deterministic: no model calls."""

from unittest.mock import patch

import pytest

from agents import protocol_analysis_agent as agent
from agents.plain_language_agent import (
    approval_block_reason,
    verification_warnings,
    approve_item,
    edit_item,
    generate_patient_guide,
    reject_item,
    reverify_item,
)
from models.protocol_models import (
    EligibilityCriterion,
    PatientInfoItem,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)
from tests.publish_helpers import GENERIC_SUMMARY
from services import analysis_cache, review_store
from services.coverage import build_coverage, coverage_block_reason
from services.guide_review import build_published_snapshot, decision_block_reason
from services.protocol_accuracy import prefer_labeled_title
from services.protocol_pipeline import analyze_protocol_pdf
from services.source_verifier import verify_source_pages
from services.summary_review import (
    change_diff,
    extractive_summary,
    generate_patient_summary,
    is_summary_approved,
    summary_approval_block_reason,
    summary_publish_block_reason,
    validate_summary,
)

PAGES = {
    1: "PROTOCOL TITLE: Oral Iron Schedules in Iron Deficiency Anemia\nPROTOCOL ID: SYN-IDA-01",
    2: (
        "ELIGIBILITY\nAdults aged 18 to 65 years with iron deficiency anemia.\n"
    ),
    4: (
        "STUDY VISITS\nScreening Visit: Day -14 to Day -1.\nBaseline Visit: Day 1.\n"
        "Follow-up 1: Week 2 ±3 days.\nFollow-up 2: Week 4 ±3 days.\n"
        "Final Visit: Week 8 ±5 days."
    ),
    5: "Blood collection occurs at five visits. About 45 mL of blood is collected in total.",
    10: "EVALUATION QUESTIONS\nWill the participant have blood glucose tested at Day 30?",
}


def _pdf():
    return {"pages": [{"page_number": n, "text": t} for n, t in PAGES.items()]}


def _visit(name, timing, page=4):
    return TrialVisit(visit_name=name, timing=timing, source_page=page)


def _correct_analysis():
    return ProtocolAnalysis(
        trial_title="Oral Iron Schedules in Iron Deficiency Anemia",
        protocol_id="SYN-IDA-01",
        analysis_complete=True,
        inclusion_criteria=[
            EligibilityCriterion(
                description="Adults aged 18 to 65 years with iron deficiency anemia.",
                source_page=2,
            )
        ],
        visits=[
            _visit("Screening Visit", "Day -14 to Day -1"),
            _visit("Baseline Visit", "Day 1"),
            _visit("Follow-up 1", "Week 2 ±3 days"),
            _visit("Follow-up 2", "Week 4 ±3 days"),
            _visit("Final Visit", "Week 8 ±5 days"),
        ],
        patient_information=[
            PatientInfoItem(
                category="blood_collection",
                description="Blood collection occurs at five visits.",
                source_page=5,
            ),
        ],
    )


def _echo(original, section, feedback=None):
    return original


def _guide(analysis, verification):
    return generate_patient_guide(analysis, rewriter=_echo, verification=verification)


def _find(guide, text):
    return next(i for i in guide.all_items() if text in i.original_text)


# --- Extraction accuracy ---------------------------------------------------


def test_five_scheduled_visits_verify_against_their_pages():
    rows = verify_source_pages(_correct_analysis(), _pdf())
    visit_rows = [r for r in rows if r["category"] == "visits"]
    assert len(visit_rows) == 5
    assert all(r["status"] == "exact_match" for r in visit_rows)
    assert {r["source_page"] for r in visit_rows} == {4}


def test_invented_day_30_visit_is_not_verified():
    analysis = _correct_analysis()
    analysis.visits.append(_visit("Follow-up Visit", "Day 30"))
    rows = verify_source_pages(analysis, _pdf())
    invented = next(r for r in rows if r["statement"] == "Follow-up Visit")
    assert invented["status"] != "exact_match"


def test_wrong_visit_timing_is_flagged_even_though_name_matches():
    analysis = _correct_analysis()
    analysis.visits[0].timing = "Day 0"
    rows = verify_source_pages(analysis, _pdf())
    assert rows[1]["status"] == "timing_not_found"


@pytest.mark.parametrize("name", ["Blood glucose testing", "Blood pressure measurement"])
def test_unsupported_procedures_never_verify(name):
    analysis = ProtocolAnalysis(
        procedures=[TrialProcedure(name=name, source_page=5)]
    )
    row = verify_source_pages(analysis, _pdf())[0]
    assert row["status"] != "exact_match"


def test_evaluation_question_text_is_not_a_protocol_fact():
    analysis = ProtocolAnalysis(
        procedures=[
            TrialProcedure(
                name="Will the participant have blood glucose tested at Day 30?",
                source_page=10,
            )
        ]
    )
    row = verify_source_pages(analysis, _pdf())[0]
    assert row["status"] == "illustrative_only"


def test_patient_information_keeps_its_source_page():
    rows = verify_source_pages(_correct_analysis(), _pdf())
    info = next(r for r in rows if r["category"] == "patient_information")
    assert info["status"] == "exact_match" and info["source_page"] == 5


def test_prompts_no_longer_contain_the_misleading_examples():
    prompts = []
    with patch("services.model_provider.ollama.Client") as client:
        client.return_value.chat.side_effect = RuntimeError("no model in tests")
        try:
            agent.analyze_protocol("[SOURCE PAGE 1]\nPROTOCOL TITLE: T\n")
        except Exception:
            pass
        prompts = [c.kwargs["messages"][0]["content"] for c in client.return_value.chat.call_args_list]
    assert prompts
    for prompt in prompts:
        head = prompt.split("PROTOCOL TEXT:")[0]
        for bad in ("blood glucose", "blood pressure", "Day 30"):
            assert bad.lower() not in head.lower()


def test_title_conflict_across_chunks_keeps_title_without_blocking():
    first = ProtocolAnalysis(trial_title="Real Title", protocol_id="P-1")
    second = ProtocolAnalysis(trial_title="Other Heading Text", protocol_id="P-1")
    merged, warnings = agent._merge_chunk_analyses([first, second])
    assert merged.trial_title == "Real Title"
    assert any("trial_title" in w for w in warnings)


def test_labeled_title_wins_over_model_title():
    analysis = ProtocolAnalysis(trial_title="Page Header Text")
    text = "[SOURCE PAGE 1]\nPROTOCOL TITLE: Oral Iron Schedules in Iron Deficiency Anemia\n"
    prefer_labeled_title(analysis, text)
    assert analysis.trial_title == "Oral Iron Schedules in Iron Deficiency Anemia"


def test_chunk_merge_conflict_does_not_mark_analysis_incomplete():
    merged, _ = agent._merge_chunk_analyses(
        [ProtocolAnalysis(trial_title="A"), ProtocolAnalysis(trial_title="B")]
    )
    assert merged.trial_title == "A"


# --- Per-statement approval ------------------------------------------------


def _guide_with_one_bad_statement():
    analysis = _correct_analysis()
    analysis.inclusion_criteria.append(
        EligibilityCriterion(description="Patients must have blood glucose measured.", source_page=2)
    )
    rows = verify_source_pages(analysis, _pdf())
    return _guide(analysis, rows)


def test_verified_statement_is_approvable_while_another_is_flagged():
    guide = _guide_with_one_bad_statement()
    good = _find(guide, "Adults aged 18 to 65")
    bad = _find(guide, "blood glucose")
    assert bad.source_status == "unverified"
    assert approval_block_reason(good) is None
    approve_item(good)
    assert good.approved


def test_unverified_statement_cannot_be_approved_and_reason_is_shown():
    guide = _guide_with_one_bad_statement()
    bad = _find(guide, "blood glucose")
    assert approval_block_reason(bad) is None  # warnings are not blockers
    warnings = verification_warnings(bad)
    assert warnings and "Not verified" in warnings[0]
    with pytest.raises(ValueError):  # approval needs an explicit override reason
        approve_item(bad)
    assert not bad.approved


def test_unsupported_statement_can_be_rejected_and_never_reaches_patients():
    guide = _guide_with_one_bad_statement()
    bad = _find(guide, "blood glucose")
    reject_item(bad)
    for item in guide.all_items():
        if item is not bad and item.source_status == "verified":
            item.approved = True
    snapshot = build_published_snapshot(guide)
    assert "blood glucose" not in snapshot.model_dump_json()


def test_unverified_statement_blocks_publication_even_if_flag_forced():
    guide = _guide_with_one_bad_statement()
    for item in guide.all_items():
        item.approved = True
    # Approval flags without a recorded override do not count as decisions.
    assert "decision" in decision_block_reason(guide)
    assert all("blood glucose" not in i.plain_text.lower()
               for i in build_published_snapshot(guide).all_items())


def test_edited_statement_needs_source_support_before_approval():
    guide = _guide_with_one_bad_statement()
    bad = _find(guide, "blood glucose")
    edit_item(bad, "Adults aged 18 to 65 years with iron deficiency anemia.")
    assert reverify_item(bad, source_page_text=PAGES[2]) == []
    assert approval_block_reason(bad) is None

    edit_item(bad, "Adults aged 18 to 70 years.")
    reverify_item(bad, source_page_text=PAGES[2])
    assert verification_warnings(bad)
    with pytest.raises(ValueError):
        approve_item(bad)


# --- Coverage / critical categories ----------------------------------------


def _pipeline_result(analysis):
    return {
        "analysis": analysis,
        "completeness_audit": {"categories": {}},
        "pages": {"total": 10, "extracted": 10, "analyzed": 10},
    }


def _approved(guide):
    for item in guide.all_items():
        if item.source_status == "verified":
            item.approved = True
    return guide


def test_publication_blocked_when_critical_category_has_no_approved_statement():
    analysis = _correct_analysis()
    guide = _approved(_guide(analysis, verify_source_pages(analysis, _pdf())))
    coverage = build_coverage(_pipeline_result(analysis), guide)
    reason = coverage_block_reason(coverage, {}, True)
    assert reason and "Cannot publish" in reason


def test_not_applicable_override_needs_reason_and_cannot_hide_extracted_content():
    analysis = _correct_analysis()
    guide = _approved(_guide(analysis, verify_source_pages(analysis, _pdf())))
    coverage = build_coverage(_pipeline_result(analysis), guide)
    overrides = {
        k: {"reason": "x"} for k in ("interventions", "risks", "consent", "procedures")
    }
    assert coverage_block_reason(coverage, overrides, True)  # reason too short
    good = {
        k: {"reason": "Checked the source: this protocol has no such section."}
        for k in ("interventions", "risks", "consent")
    }
    assert coverage_block_reason(coverage, good, True) is None
    # Visits were extracted, so they cannot be waived.
    waived = {"visits": {"reason": "Checked the source: nothing to report here."}}
    for item in guide.visits.items:
        item.approved = False
    coverage = build_coverage(_pipeline_result(analysis), guide)
    assert "Study visits" in coverage_block_reason(coverage, {**good, **waived}, True)


def test_override_refused_when_analysis_incomplete():
    analysis = _correct_analysis()
    guide = _approved(_guide(analysis, verify_source_pages(analysis, _pdf())))
    coverage = build_coverage(_pipeline_result(analysis), guide)
    good = {"consent": {"reason": "Checked the source: this protocol has none."}}
    assert coverage_block_reason(coverage, good, False)


def test_pages_extracted_analyzed_and_statements_are_reported_separately():
    analysis = _correct_analysis()
    guide = _guide(analysis, verify_source_pages(analysis, _pdf()))
    coverage = build_coverage(
        {"analysis": analysis, "pages": {"total": 10, "extracted": 9, "analyzed": 7}},
        guide,
    )
    assert coverage["pages"]["extracted"] == 9 and coverage["pages"]["analyzed"] == 7


# --- Summary editing, validation, approval, audit --------------------------


def _summary_guide():
    analysis = _correct_analysis()
    guide = _approved(_guide(analysis, verify_source_pages(analysis, _pdf())))
    return guide


def test_summary_generation_uses_writer_and_falls_back_deterministically():
    guide = _summary_guide()
    text, source = generate_patient_summary(guide, writer=lambda p: "")
    assert text and source != "llm"
    assert extractive_summary(guide)


def test_summary_validation_catches_changed_numbers_and_invented_tests():
    guide = _summary_guide()
    assert any("45" in p or "number" in p.lower() for p in validate_summary("You attend 3 visits over 45 weeks.", guide))
    problems = validate_summary("Your blood glucose is checked every visit.", guide)
    assert problems


def test_summary_edit_audit_and_approval_invalidation():
    pid = "protocol-summary"
    review_store.save_original_summary(pid, "AI original text.", "llm")
    assert review_store.load_summary(pid)["original_text"] == "AI original text."
    assert review_store.save_summary_draft(pid, "Coordinator edit one.")
    record = review_store.load_summary(pid)
    assert record["original_text"] == "AI original text."
    assert record["draft_text"] == "Coordinator edit one."

    assert review_store.approve_summary(pid)
    assert is_summary_approved(review_store.load_summary(pid))

    review_store.save_summary_draft(pid, "Coordinator edit two.")
    record = review_store.load_summary(pid)
    assert not is_summary_approved(record)
    assert record["original_text"] == "AI original text."

    actions = [row["action"] for row in review_store.summary_audit(pid)]
    assert len(actions) >= 4
    assert any("approv" in a for a in actions)
    assert any("invalid" in a or "draft" in a for a in actions)


def test_unapproved_or_stale_summary_blocks_publication():
    guide = _summary_guide()
    snapshot = build_published_snapshot(guide)
    pid = "protocol-stale"
    assert summary_publish_block_reason(None, snapshot)
    review_store.save_original_summary(pid, extractive_summary(guide), "llm")
    assert summary_publish_block_reason(review_store.load_summary(pid), snapshot)
    review_store.approve_summary(pid)
    assert summary_publish_block_reason(review_store.load_summary(pid), snapshot) is None
    review_store.save_summary_draft(pid, "A different summary.")
    assert summary_publish_block_reason(review_store.load_summary(pid), snapshot)


def test_summary_with_invalid_content_cannot_be_approved():
    guide = _summary_guide()
    pid = "protocol-bad-summary"
    review_store.save_original_summary(pid, "Fine summary.", "llm")
    review_store.save_summary_draft(pid, "You will have blood glucose and blood pressure checks.")
    assert summary_approval_block_reason(review_store.load_summary(pid), guide)


def test_change_diff_shows_edits():
    assert change_diff("one\ntwo", "one\nthree")


# --- Cache behaviour -------------------------------------------------------


def test_cache_key_changes_with_analysis_version_and_text():
    settings = {"model": "m", "version": agent.ANALYSIS_VERSION}
    key = analysis_cache.make_key("text", settings)
    assert analysis_cache.make_key("text", {**settings, "version": "old"}) != key
    assert analysis_cache.make_key("changed", settings) != key


def test_cold_then_warm_cache_without_model_calls():
    pages = [
        {"page_number": 1, "text": "PROTOCOL TITLE: Cache Study\nParticipants attend visits."},
        {"page_number": 2, "text": "INCLUSION CRITERIA\nAdults 18 to 65."},
    ]
    analysis = ProtocolAnalysis(
        trial_title="Cache Study",
        analysis_complete=True,
        inclusion_criteria=[EligibilityCriterion(description="Adults 18 to 65.", source_page=2)],
    )
    with patch(
        "services.protocol_pipeline.extract_pdf",
        return_value={"pages": pages, "warnings": []},
    ), patch("services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS", 10), patch(
        "services.protocol_pipeline.analyze_protocol_in_chunks",
        return_value=(analysis, [], [], [], 2),
    ) as run:
        cold = analyze_protocol_pdf("cache.pdf")
        warm = analyze_protocol_pdf("cache.pdf")
        assert run.call_count == 1
        assert warm["llm_call_count"] == 0
        # Verification still re-runs on a cache hit and reports the new fields.
        assert warm["verification_passed"] is True and warm["execution_complete"] is True
        assert warm["pages"]["total"] == 2 and cold["pages"]["total"] == 2

    with patch.object(agent, "ANALYSIS_VERSION", "different"), patch(
        "services.protocol_pipeline.extract_pdf",
        return_value={"pages": pages, "warnings": []},
    ), patch("services.protocol_pipeline.LONG_PROTOCOL_THRESHOLD_CHARS", 10), patch(
        "services.protocol_pipeline.analyze_protocol_in_chunks",
        return_value=(analysis, [], [], [], 2),
    ) as run:
        analyze_protocol_pdf("cache.pdf")
        assert run.call_count == 1  # new version => cold run


# --- Portal shows only the approved summary --------------------------------


def test_published_snapshot_carries_approved_summary_not_the_draft():
    from tests.test_patient_guide_ui import (
        _approved_guide,
        _click,
        _eight_statement_result,
        _prepare_review_app,
    )

    result = _eight_statement_result()
    at = _prepare_review_app(result, _approved_guide(result))  # approves GENERIC_SUMMARY
    review_store.save_summary_draft("protocol-eight", "UNREVIEWED DRAFT TEXT")
    at = at.run()
    assert at.button(key="publish_protocol_btn").disabled  # draft not approved

    review_store.save_summary_draft("protocol-eight", GENERIC_SUMMARY)
    review_store.approve_summary("protocol-eight")
    at = _click(at.run(), "publish_protocol_btn")
    published = at.session_state["published_protocols"]["protocol-eight"]["guide"]
    assert published.summary == GENERIC_SUMMARY
    assert "UNREVIEWED" not in published.model_dump_json()


# --- Safety review: legacy records, reverify evidence, visit binding -------

import json

from agents.plain_language_agent import approve_all_unflagged, recheck_sources
from models.plain_language_models import GuideItem, PatientGuide
from services.guide_review import patient_visible_guide
from services.source_verifier import unsupported_claims
from ui.patient_guide_tab import is_fully_approved_snapshot


def _legacy_guide():
    """A guide saved before source checks existed (no source_status in JSON)."""
    analysis = _correct_analysis()
    guide = _guide(analysis, verify_source_pages(analysis, _pdf()))
    data = json.loads(guide.model_dump_json())
    for section in ("who_can_join", "who_cannot_join", "visits", "procedures",
                    "safety", "interventions", "consent"):
        for item in data[section]["items"]:
            item.pop("source_status", None)
            item.pop("source_issue", None)
            item["approved"] = True
    return PatientGuide.model_validate(data), analysis


def test_legacy_items_load_unchecked_not_verified():
    guide, _ = _legacy_guide()
    items = guide.all_items()
    assert items and all(i.source_status == "unchecked" for i in items)
    assert all(i.approved for i in items)  # saved work is kept


def test_legacy_items_cannot_be_approved_published_or_bulk_approved():
    guide, _ = _legacy_guide()
    item = guide.visits.items[0]
    item.approved = False
    assert "Not yet checked" in verification_warnings(item)[0]
    with pytest.raises(ValueError):
        approve_item(item)
    assert approve_all_unflagged(guide) == 0
    item.approved = True  # even a forced approval flag
    assert "decision" in decision_block_reason(guide).lower()
    assert build_published_snapshot(guide).all_items() == []


def test_fresh_source_check_resolves_legacy_items_and_keeps_decisions():
    guide, analysis = _legacy_guide()
    analysis.visits[0].timing = "Day 0"  # legacy data contained a wrong timing
    rows = verify_source_pages(analysis, _pdf())
    assert recheck_sources(guide, rows) > 0
    by_name = {i.original_text.split(".")[0]: i for i in guide.visits.items}
    good = by_name["Baseline Visit"]
    bad = by_name["Screening Visit"]
    assert good.source_status == "verified" and good.approved
    assert bad.source_status == "unverified" and not bad.approved
    assert bad.plain_text  # wording is not deleted


def test_recheck_without_rows_leaves_items_unchecked():
    guide, _ = _legacy_guide()
    assert recheck_sources(guide, []) == 0
    assert all(i.source_status == "unchecked" for i in guide.all_items())


def _unverified_visit():
    analysis = _correct_analysis()
    analysis.visits[0].timing = "Day 0"
    guide = _guide(analysis, verify_source_pages(analysis, _pdf()))
    return next(i for i in guide.visits.items if "Screening" in i.original_text)


def test_edit_with_page_numbers_but_unsupported_meaning_is_not_verified():
    item = _unverified_visit()
    assert item.source_status == "unverified"
    # 14, 1 and 2 all occur on page 4, but this relationship is not stated.
    edit_item(item, "Screening Visit happens at Week 2 and Day -14.")
    assert reverify_item(item, source_page_text=PAGES[4])
    assert item.source_status == "unverified"
    assert verification_warnings(item)


def test_edit_matching_the_protocol_passes_and_keeps_the_replaced_text():
    item = _unverified_visit()
    before = item.original_text
    edit_item(item, "Screening Visit: Day -14 to Day -1.")
    assert reverify_item(item, source_page_text=PAGES[4]) == []
    assert item.source_status == "verified"
    assert item.superseded_text == before
    assert approval_block_reason(item) is None


def test_edit_cannot_be_verified_by_example_text_or_without_page_text():
    item = _unverified_visit()
    edit_item(item, "Follow-up Visit: Day 30.")
    assert reverify_item(item, source_page_text=PAGES[10])
    assert reverify_item(item, source_page_text=None)
    assert item.source_status == "unverified"


def test_negation_and_numbers_must_come_from_the_same_passage():
    page = "Take one tablet daily.\nDo not take it with milk. 5 mL is drawn at Week 2."
    assert unsupported_claims("Take one tablet daily.", page) == []
    assert unsupported_claims("Take it with milk.", page)  # drops the negation
    assert unsupported_claims("One tablet daily at Week 2.", page)  # figures split


def test_timing_must_be_written_with_the_visit_not_just_on_the_page():
    pdf = {"pages": [{"page_number": 1, "text": (
        "Screening Visit: Day -14 to Day -1\nBaseline Visit: Day 1\n"
        "Final Visit: Week 8"
    )}]}
    swapped = ProtocolAnalysis(visits=[
        TrialVisit(visit_name="Screening Visit", timing="Week 8", source_page=1),
        TrialVisit(visit_name="Final Visit", timing="Day 1", source_page=1),
    ])
    rows = verify_source_pages(swapped, pdf)
    assert [r["status"] for r in rows] == ["timing_not_found", "timing_not_found"]
    ok = ProtocolAnalysis(visits=[
        TrialVisit(visit_name="Final Visit", timing="Week 8", source_page=1)
    ])
    assert verify_source_pages(ok, pdf)[0]["status"] == "exact_match"


def test_table_wrapped_timing_is_accepted_only_when_unambiguous():
    pdf = {"pages": [{"page_number": 1, "text": (
        "Follow-up 1\nWeek 2 ±3 days\nFollow-up 2\nWeek 4 ±3 days"
    )}]}
    analysis = ProtocolAnalysis(visits=[
        TrialVisit(visit_name="Follow-up 1", timing="Week 2 ±3 days", source_page=1),
        TrialVisit(visit_name="Follow-up 1", timing="Week 4 ±3 days", source_page=1),
    ])
    assert [r["status"] for r in verify_source_pages(analysis, pdf)] == [
        "exact_match", "timing_not_found",
    ]


def test_portal_hides_unverified_and_snapshot_validator_rejects_them():
    guide = _summary_guide()
    victim = guide.visits.items[0]
    victim.source_status = "unverified"
    shown = patient_visible_guide(guide)
    assert victim.original_text not in shown.model_dump_json()
    assert not is_fully_approved_snapshot(guide)


def test_publish_block_reason_without_protocol_id_or_audit_cannot_be_bypassed(monkeypatch):
    import streamlit as st
    from app import publish_block_reason

    analysis = _correct_analysis()
    guide = _approved(_guide(analysis, verify_source_pages(analysis, _pdf())))
    result = {**_pipeline_result(analysis), "analysis_complete": True}
    monkeypatch.setattr(st, "session_state", {})
    assert "no active identifier" in publish_block_reason(guide, result)

    # No audit => "Not applicable" cannot be trusted.
    no_audit = {k: v for k, v in result.items() if k != "completeness_audit"}
    coverage = build_coverage(no_audit, guide)
    overrides = {"interventions": {"reason": "Checked the source: none exists."}}
    assert coverage_block_reason(coverage, overrides, True)


def test_coverage_counts_only_verified_approved_statements():
    analysis = _correct_analysis()
    guide = _approved(_guide(analysis, verify_source_pages(analysis, _pdf())))
    for item in guide.visits.items:
        item.source_status = "unchecked"
    entry = build_coverage(_pipeline_result(analysis), guide)["categories"]["visits"]
    assert entry["approved"] == 0
