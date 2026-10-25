"""
Regression tests for the final accuracy fixes, built on the 50-page benchmark
(one synopsis block repeated on every page). No model is ever called.
"""

import threading
import time

from agents.plain_language_agent import generate_patient_guide, statement_specs
from models.plain_language_models import GuideItem, GuideSection, PatientGuide, page_label
from models.protocol_models import (
    EligibilityCriterion, PatientInfoItem, ProtocolAnalysis, SafetyInformation,
    TrialVisit as Visit,
)
from services.protocol_accuracy import (
    apply_protocol_accuracy_checks, normalize_visit_labels, recover_labeled_sections,
)
from services.source_verifier import verify_source_pages
from services.summary_review import (
    _generate_patient_summary, extractive_summary, summary_timeout_seconds,
    validate_summary,
)

SYNOPSIS = (
    "001 Protocol synopsis\n\nStudy purpose\n"
    "A fictional 24-week randomized pilot comparing standard education with optional "
    "app-supported glucose logging for adults\nwith type 2 diabetes.\n\n"
    "Eligibility - potential inclusion\nAdults aged 30-75 with established type 2 diabetes.\n\n"
    "Study timeline\nPlanned visits: Screening, Baseline, Weeks 2, 4, 8, 12, 18, and 24. "
    "Visit dates are approximate and confirmed by the study\ncoordinator.\n\n"
    "Participant rights\nNo study tool changes medication doses.\n"
)


def pdf(*texts):
    return {"pages": [{"page_number": i, "text": t} for i, t in enumerate(texts, 1)]}


def analysis(**kw):
    return ProtocolAnalysis(trial_title="T", protocol_id="P", **kw)


# ---------------------------------------------------------------- Fix 1

def test_study_purpose_is_recovered_verbatim_with_every_page():
    a = analysis()
    recover_labeled_sections(a, pdf(SYNOPSIS, "010 Other\n\n" + SYNOPSIS))
    purpose = [i for i in a.patient_information if i.category == "study_purpose"]
    assert [i.source_page for i in purpose] == [1, 2]
    assert all("24-week randomized pilot" in i.description for i in purpose)


def test_page_title_without_text_adds_nothing_and_nothing_is_invented():
    a = analysis()
    recover_labeled_sections(a, pdf("016 Participant withdrawal\n\nProcedures\nBlood draw."))
    assert a.patient_information == [] and a.safety_information == []


def test_consent_withdrawal_and_safety_monitoring_are_recovered_from_their_sections():
    text = (
        "Informed consent\nYou must sign a consent form before any study activity.\n\n"
        "Withdrawal\nYou may leave the study at any time.\n\n"
        "Safety monitoring\nStudy staff review symptoms at every visit.\n"
    )
    a = analysis()
    recover_labeled_sections(a, pdf(text))
    consent = [i.description for i in a.patient_information if i.category == "consent_withdrawal"]
    assert "You may leave the study at any time." in consent
    assert any("consent form" in c for c in consent)
    assert [s.description for s in a.safety_information] == [
        "Study staff review symptoms at every visit."
    ]
    assert all(i.source_page == 1 for i in a.patient_information)


def test_already_extracted_statements_are_not_added_twice():
    a = analysis(patient_information=[PatientInfoItem(
        category="study_purpose", source_page=1,
        description="A fictional 24-week randomized pilot comparing standard education "
                    "with optional app-supported glucose logging for adults with type 2 diabetes",
    )])
    recover_labeled_sections(a, pdf(SYNOPSIS))
    assert len([i for i in a.patient_information if i.category == "study_purpose"]) == 1


def test_example_sections_are_never_recovered():
    a = analysis()
    recover_labeled_sections(
        a, pdf("Withdrawal\nYou may leave.\n\nREVIEW QUESTIONS\nStudy purpose\nA made up purpose here.")
    )
    assert all("made up" not in i.description for i in a.patient_information)


def test_purpose_reaches_the_patient_guide_section_with_pages():
    a = analysis()
    recover_labeled_sections(a, pdf(SYNOPSIS, SYNOPSIS))
    guide = generate_patient_guide(a, rewriter=lambda t, s, feedback=None: t)
    assert len(guide.purpose.items) == 1
    assert guide.purpose.items[0].source_pages == [2]


# ---------------------------------------------------------------- Fix 2

def test_identical_statements_on_many_pages_become_one_with_all_pages():
    a = analysis(inclusion_criteria=[
        EligibilityCriterion(description="Adults aged 30-75", source_page=p)
        for p in (1, 5, 9)
    ] + [EligibilityCriterion(description="Adults aged 30-75.", source_page=13)])
    specs = statement_specs(a)
    assert len(specs) == 1
    assert specs[0].page == 1 and specs[0].pages == (1, 5, 9, 13)


def test_materially_different_statements_are_not_merged():
    a = analysis(inclusion_criteria=[
        EligibilityCriterion(description="Adults aged 30-75", source_page=1),
        EligibilityCriterion(description="Adults aged 30-65", source_page=5),
    ])
    assert len(statement_specs(a)) == 2


def test_visits_merge_by_name_but_conflicting_timings_stay_separate():
    a = analysis(visits=[
        Visit(visit_name="Week 2", timing="approximate", source_page=1),
        Visit(visit_name="Week 2", timing=None, source_page=10),
        Visit(visit_name="Week 2", timing="approximate", source_page=20),
    ])
    specs = statement_specs(a)
    assert len(specs) == 1 and specs[0].pages == (1, 10, 20)
    assert "Timing: approximate" in specs[0].text

    conflicting = analysis(visits=[
        Visit(visit_name="Week 2", timing="Week 2 ±3 days", source_page=1),
        Visit(visit_name="Week 2", timing="Week 3 ±3 days", source_page=2),
    ])
    assert len(statement_specs(conflicting)) == 2


def test_merged_pages_are_shown_and_not_lost_in_the_guide():
    a = analysis(inclusion_criteria=[
        EligibilityCriterion(description="Adults aged 30-75", source_page=p) for p in (1, 5)
    ])
    guide = generate_patient_guide(a, rewriter=lambda t, s, feedback=None: t)
    item = guide.who_can_join.items[0]
    assert item.source_page == 1 and item.source_pages == [5]
    assert page_label(item) == "pages 1, 5"
    assert page_label(GuideItem(section="visits", original_text="x", plain_text="x",
                                source_page=3)) == "page 3"


# ---------------------------------------------------------------- Fix 3

def _visits(*names, timing="approximate", page=1):
    return [Visit(visit_name=n, timing=timing, source_page=page) for n in names]


def test_week_list_visits_verify_against_the_planned_visit_list():
    a = analysis(visits=_visits("Week 12", "Weeks 18", "Week 24", "Week 2"))
    rows = verify_source_pages(a, pdf(SYNOPSIS))
    assert [r["status"] for r in rows] == ["exact_match"] * 4


def test_week_not_in_the_list_is_still_unverified_and_never_invented():
    a = analysis(visits=_visits("Week 3", "Day 30"))
    rows = verify_source_pages(a, pdf(SYNOPSIS))
    assert [r["status"] for r in rows] == ["not_verified", "not_verified"]


def test_timing_must_come_from_the_week_list_sentence():
    a = analysis(visits=_visits("Week 4", timing="±3 days"))
    assert verify_source_pages(a, pdf(SYNOPSIS))[0]["status"] == "timing_not_found"


def test_labels_normalized_and_coordinator_confirmation_preserved():
    a = analysis(visits=_visits("Weeks 2", "Weeks 4", "Screening"))
    warnings = normalize_visit_labels(a, pdf(SYNOPSIS))
    assert [v.visit_name for v in a.visits] == ["Week 2", "Week 4", "Screening"]
    assert all(v.timing == "approximate and confirmed by the study coordinator"
               for v in a.visits)
    assert warnings
    rows = verify_source_pages(a, pdf(SYNOPSIS))
    assert [r["status"] for r in rows[:2]] == ["exact_match", "exact_match"]


def test_timing_is_left_alone_when_the_page_has_no_matching_sentence():
    a = analysis(visits=[Visit(visit_name="Week 2", timing="±3 days", source_page=1)])
    normalize_visit_labels(a, pdf(SYNOPSIS))
    assert a.visits[0].timing == "±3 days"


def test_full_accuracy_pass_yields_verified_unique_visits():
    a = analysis(visits=_visits("Weeks 2", "Weeks 4", "Weeks 12") +
                        _visits("Weeks 2", timing=None, page=2))
    source = pdf(SYNOPSIS, SYNOPSIS)
    apply_protocol_accuracy_checks(a, source)
    assert [v.visit_name for v in a.visits][:3] == ["Week 2", "Week 4", "Week 12"]
    assert all(r["status"] == "exact_match" for r in verify_source_pages(a, source))
    assert "Day 30" not in " ".join(s.text for s in statement_specs(a))


# ---------------------------------------------------------------- Fix 4

def _guide():
    def item(text, section):
        return GuideItem(section=section, original_text=text, plain_text=text,
                         source_page=1, source_status="verified", approved=False)
    return PatientGuide(
        trial_title="T", disclaimer="d",
        who_can_join=GuideSection(title="Who may be able to join", introduction="", items=[
            item("Adults aged 30-75 can be considered.", "inclusion_criteria")]),
        who_cannot_join=GuideSection(title="Who may not be able to join", introduction=""),
        visits=GuideSection(title="Trial visits", introduction=""),
        procedures=GuideSection(title="Tests and procedures", introduction=""),
        safety=GuideSection(title="Safety information", introduction=""),
    )


def test_stalled_summary_model_falls_back_within_the_timeout(monkeypatch):
    monkeypatch.setenv("CTAI_SUMMARY_TIMEOUT_SECONDS", "0.2")
    release = threading.Event()
    guide = _guide()
    started = time.perf_counter()
    text, by = _generate_patient_summary(guide, lambda prompt: release.wait(30) or "late")
    elapsed = time.perf_counter() - started
    release.set()
    assert by == "fallback" and text == extractive_summary(guide)
    assert elapsed < 2  # one stalled call, no second attempt
    assert validate_summary(text, guide) == []


def test_model_error_falls_back_and_fast_model_is_unaffected(monkeypatch):
    monkeypatch.setenv("CTAI_SUMMARY_TIMEOUT_SECONDS", "5")
    guide = _guide()

    def boom(prompt):
        raise RuntimeError("model down")

    assert _generate_patient_summary(guide, boom)[1] == "fallback"
    good = extractive_summary(guide)
    assert _generate_patient_summary(guide, lambda p: good) == (good, "llm")


def test_summary_timeout_setting_is_configurable_with_safe_default(monkeypatch):
    monkeypatch.delenv("CTAI_SUMMARY_TIMEOUT_SECONDS", raising=False)
    assert summary_timeout_seconds() == 120.0
    for bad in ("abc", "0", "-5", "nan", "inf"):
        monkeypatch.setenv("CTAI_SUMMARY_TIMEOUT_SECONDS", bad)
        assert summary_timeout_seconds() == 120.0
    monkeypatch.setenv("CTAI_SUMMARY_TIMEOUT_SECONDS", "45")
    assert summary_timeout_seconds() == 45.0


def test_timed_out_summary_is_still_unapproved_and_not_in_the_portal():
    guide = _guide()
    assert guide.summary is None  # nothing is published until a coordinator approves
    assert not guide.who_can_join.items[0].approved


def test_screening_and_baseline_verify_from_the_planned_visit_list():
    a = analysis(visits=_visits("Screening", "Baseline", "Final visit"))
    rows = verify_source_pages(a, pdf(SYNOPSIS))
    assert [r["status"] for r in rows] == ["exact_match", "exact_match", "not_verified"]


def test_listed_weeks_raise_no_unsupported_warning_but_invented_ones_do():
    a = analysis(visits=_visits("Weeks 4", "Weeks 30"))
    warnings = apply_protocol_accuracy_checks(a, pdf(SYNOPSIS))
    assert not any("Week 4" in w and "unsupported" in w for w in warnings)
    assert any("30" in w and "unsupported" in w for w in warnings)


# Found in the real 2-page Ollama end-to-end run: details the protocol never gave.
def test_invented_test_location_blood_draw_and_device_are_flagged():
    from services import grounding
    support = "Blood glucose testing.\nBlood pressure measurement."
    for invented in (
        "Blood glucose testing is done at the clinic.",
        "You will have your blood drawn to check your blood sugar level.",
        "This is done using a device that checks your blood pressure.",
    ):
        assert grounding.unsupported_terms(invented, support), invented
    assert not grounding.unsupported_terms("Your blood glucose will be tested.", support)
    assert not grounding.unsupported_terms(
        "A blood sample is taken at the clinic.", "Blood sample collected at the clinic."
    )


def test_fallback_summary_blocks_end_in_full_stops():
    guide = _guide()
    guide.visits.items.append(GuideItem(
        section="visits", original_text="Screening Visit. Timing: Day 0",
        plain_text="Screening Visit. Timing: Day 0", source_page=2,
        source_status="verified",
    ))
    guide.visits.items.append(GuideItem(
        section="visits", original_text="Follow-up", plain_text="A follow-up visit on Day 30.",
        source_page=2, source_status="verified",
    ))
    assert "Day 0. A follow-up" in extractive_summary(guide)
