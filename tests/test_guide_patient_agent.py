"""
Tests for the Patient Assistant that answers only from the approved guide.

Covers the integration issue found in review: when only the age requirement
was approved, the Patient Assistant still showed the unapproved "Type 2
diabetes" requirement because it read the full protocol analysis.
"""

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agents.guide_patient_agent import APPROVED_STATUS, answer_patient_from_guide
from agents.plain_language_agent import approve_item, generate_patient_guide
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialProcedure,
    TrialVisit,
)

AGE = "Participants must be between 18 and 65 years old."
DIABETES = "Participants must have a diagnosis of Type 2 diabetes."
DIABETES_DURATION = (
    "Participants must have a diagnosis of Type 2 diabetes for at least 2 years."
)
EGFR = "eGFR below 45 mL/min/1.73 m2."
HYPOGLYCEMIA = "Risk of hypoglycemia."
BLOOD_PRESSURE = "Blood pressure measurement."


def _fake_rewriter(original, section, feedback=None):
    return f"In simple words: {original}"


def _analysis():
    return ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        protocol_id="T2D-001",
        inclusion_criteria=[
            EligibilityCriterion(description=AGE, source_page=1),
            EligibilityCriterion(description=DIABETES, source_page=1),
        ],
        exclusion_criteria=[EligibilityCriterion(description=EGFR, source_page=1)],
        visits=[TrialVisit(visit_name="Follow-up Visit", timing="Day 30", source_page=2)],
        safety_information=[
            SafetyInformation(
                description="Participants will be monitored for adverse events.",
                source_page=2,
            )
        ],
    )


def _guide():
    return generate_patient_guide(_analysis(), rewriter=_fake_rewriter)


def _approve_age_only(guide):
    age_item = next(i for i in guide.who_can_join.items if "18 and 65" in i.original_text)
    approve_item(age_item)
    return guide


def test_incomplete_guide_blocks_patient_answers_even_if_item_preapproved():
    guide = _guide()
    guide.analysis_complete = False
    item = guide.who_can_join.items[0]
    item.approved = True

    answer = answer_patient_from_guide("What are the inclusion criteria?", guide)

    assert answer.intent == "not_yet_approved"
    assert "incomplete" in answer.answer_text
    assert AGE not in answer.answer_text


# ---------------------------------------------------------------------------
# The reported issue
# ---------------------------------------------------------------------------

def test_only_approved_inclusion_items_are_shared():
    guide = _approve_age_only(_guide())

    answer = answer_patient_from_guide("Who can join this study?", guide)

    assert "18 and 65" in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    assert "still being reviewed" in answer.answer_text
    assert answer.confidence == "partial"
    assert all(c.verification_status == APPROVED_STATUS for c in answer.citations)


def test_eligibility_question_uses_only_approved_criteria():
    guide = _approve_age_only(_guide())

    answer = answer_patient_from_guide("Am I eligible?", guide)

    assert answer.intent == "eligibility_decision_refused"
    assert "can't tell you whether you personally are eligible" in answer.answer_text
    assert "18 and 65" in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    assert "eGFR" not in answer.answer_text


def test_unapproved_topic_says_it_is_under_review():
    guide = _approve_age_only(_guide())

    answer = answer_patient_from_guide("Who cannot join?", guide)

    assert "still reviewing" in answer.answer_text
    assert "eGFR" not in answer.answer_text
    assert answer.citations == []


def test_nothing_approved_means_nothing_shared():
    guide = _guide()

    answer = answer_patient_from_guide("Who can join this study?", guide)

    assert answer.intent == "not_yet_approved"
    assert "18 and 65" not in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text


def test_no_guide_means_nothing_shared():
    answer = answer_patient_from_guide("What visits are there?", None)

    assert answer.intent == "not_yet_approved"
    assert "Day 30" not in answer.answer_text


# ---------------------------------------------------------------------------
# Other behaviour
# ---------------------------------------------------------------------------

def test_fully_approved_section_answers_with_page():
    guide = _guide()
    for item in guide.visits.items:
        approve_item(item)

    answer = answer_patient_from_guide("What visits will I have?", guide)

    assert answer.confidence == "high"
    assert "Follow-up Visit" in answer.answer_text
    assert "(protocol page 2)" in answer.answer_text


def test_medical_advice_is_refused_even_without_a_guide():
    answer = answer_patient_from_guide("Should I stop taking my metformin?", None)

    assert answer.intent == "medical_advice_refused"


def test_unsupported_question_offers_topics():
    guide = _approve_age_only(_guide())

    answer = answer_patient_from_guide("What is the weather today?", guide)

    assert answer.intent == "unsupported"
    assert "who may be able to join" in answer.answer_text


# ---------------------------------------------------------------------------
# Natural-language questions about specific requirements
# ---------------------------------------------------------------------------

def _approve_age_and_diabetes(guide):
    for item in guide.who_can_join.items:
        if "18 and 65" in item.original_text or "Type 2 diabetes" in item.original_text:
            approve_item(item)
    return guide


def test_approved_diabetes_requirement_question():
    guide = _approve_age_and_diabetes(_guide())

    answer = answer_patient_from_guide(
        "Do I need to have Type 2 diabetes to join this study?",
        guide,
    )

    assert answer.intent == "eligibility_inclusion"
    assert "Type 2 diabetes" in answer.answer_text
    assert "(protocol page 1)" in answer.answer_text
    assert all(c.verification_status == APPROVED_STATUS for c in answer.citations)
    # Specific question should not dump unrelated approved criteria.
    assert "18 and 65" not in answer.answer_text
    assert answer.confidence in {"high", "partial"}


def test_unapproved_diabetes_requirement_stays_hidden():
    guide = _approve_age_only(_guide())

    answer = answer_patient_from_guide(
        "Do I need to have Type 2 diabetes to join this study?",
        guide,
    )

    assert answer.intent == "eligibility_inclusion"
    assert "still reviewing" in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    assert DIABETES not in answer.answer_text
    assert answer.citations == []


def test_age_requirement_natural_question():
    guide = _approve_age_only(_guide())

    answer = answer_patient_from_guide(
        "What is the age requirement to join?",
        guide,
    )

    assert answer.intent == "eligibility_inclusion"
    assert "18 and 65" in answer.answer_text
    assert "(protocol page 1)" in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text


def test_personal_eligibility_still_refused_for_natural_phrasing():
    guide = _approve_age_and_diabetes(_guide())

    answer = answer_patient_from_guide(
        "Am I allowed to participate if I am 40?",
        guide,
    )

    assert answer.intent == "eligibility_decision_refused"
    assert "can't tell you whether you personally are eligible" in answer.answer_text
    assert "Type 2 diabetes" in answer.answer_text  # approved general criteria OK
    assert "decide" in answer.answer_text.casefold() or "study team" in answer.answer_text


def test_natural_visit_question_uses_approved_items():
    guide = _guide()
    for item in guide.visits.items:
        approve_item(item)

    answer = answer_patient_from_guide(
        "How many appointments will I have?",
        guide,
    )

    assert answer.intent == "visits"
    assert "Follow-up Visit" in answer.answer_text
    assert "(protocol page 2)" in answer.answer_text


# ---------------------------------------------------------------------------
# Blocking-issue regressions (specific no-match + age vs duration)
# ---------------------------------------------------------------------------

def _mixed_analysis():
    return ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        protocol_id="T2D-001",
        inclusion_criteria=[
            EligibilityCriterion(description=AGE, source_page=1),
            EligibilityCriterion(description=DIABETES_DURATION, source_page=1),
        ],
        exclusion_criteria=[EligibilityCriterion(description=EGFR, source_page=1)],
        visits=[TrialVisit(visit_name="Follow-up Visit", timing="Day 30", source_page=2)],
        procedures=[TrialProcedure(name=BLOOD_PRESSURE, source_page=2)],
        safety_information=[
            SafetyInformation(description=HYPOGLYCEMIA, source_page=2)
        ],
    )


def _mixed_guide():
    return generate_patient_guide(_mixed_analysis(), rewriter=_fake_rewriter)


def test_specific_question_with_no_item_match_does_not_dump_section():
    """Specific clinical cue must not fall through to an unrelated section dump."""
    guide = _mixed_guide()
    for item in guide.safety.items:
        approve_item(item)
    for item in guide.who_can_join.items:
        if "18 and 65" in item.original_text:
            approve_item(item)

    answer = answer_patient_from_guide("Is diabetes dangerous?", guide)

    assert answer.intent == "safety"
    assert "hypoglycemia" not in answer.answer_text.casefold()
    assert "don't have approved information" in answer.answer_text.casefold()
    assert answer.citations == []
    assert "Type 2 diabetes" not in answer.answer_text


def test_age_question_does_not_match_diabetes_duration_years():
    """Bare disease-duration 'years' must not be treated as an age requirement."""
    guide = _mixed_guide()
    for item in guide.who_can_join.items:
        approve_item(item)

    answer = answer_patient_from_guide(
        "What is the age requirement to join?",
        guide,
    )

    assert answer.intent == "eligibility_inclusion"
    assert "18 and 65" in answer.answer_text
    assert "(protocol page 1)" in answer.answer_text
    assert "2 years" not in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    assert answer.citations
    assert all("18 and 65" in c.statement for c in answer.citations)


def test_age_question_ignores_unapproved_diabetes_duration_match():
    """Age Q with age approved and diabetes-duration unapproved: age only, no leak."""
    guide = _mixed_guide()
    age_item = next(i for i in guide.who_can_join.items if "18 and 65" in i.original_text)
    approve_item(age_item)

    answer = answer_patient_from_guide(
        "What is the age requirement to join?",
        guide,
    )

    assert "18 and 65" in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    assert "2 years" not in answer.answer_text
    assert DIABETES_DURATION not in answer.answer_text
    # Must not claim unrelated pending "related details" from a false years match.
    assert "related details are still being reviewed" not in answer.answer_text


def test_unapproved_diabetes_duration_question_stays_hidden():
    guide = _mixed_guide()
    age_item = next(i for i in guide.who_can_join.items if "18 and 65" in i.original_text)
    approve_item(age_item)

    answer = answer_patient_from_guide(
        "Do I need to have Type 2 diabetes to join this study?",
        guide,
    )

    assert answer.intent == "eligibility_inclusion"
    assert "still reviewing" in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    assert "2 years" not in answer.answer_text
    assert answer.citations == []


def _numeric_range_analysis():
    return ProtocolAnalysis(
        trial_title="Mixed Numeric Criteria Study",
        protocol_id="NUM-1",
        inclusion_criteria=[
            EligibilityCriterion(description=AGE, source_page=1),
            EligibilityCriterion(
                description="BMI must be between 18 and 30.",
                source_page=1,
            ),
            EligibilityCriterion(
                description="Participants must take between 2 and 4 tablets daily.",
                source_page=1,
            ),
            EligibilityCriterion(description=DIABETES_DURATION, source_page=1),
        ],
    )


def test_age_question_ignores_bmi_dosage_and_duration_ranges():
    """Age matching must require age language, not bare between-N-and-M ranges."""
    guide = generate_patient_guide(
        _numeric_range_analysis(),
        rewriter=_fake_rewriter,
    )
    for item in guide.who_can_join.items:
        approve_item(item)

    answer = answer_patient_from_guide(
        "What is the age requirement to join?",
        guide,
    )

    assert answer.intent == "eligibility_inclusion"
    assert "18 and 65" in answer.answer_text
    assert "years old" in answer.answer_text
    assert "(protocol page 1)" in answer.answer_text
    assert "BMI" not in answer.answer_text
    assert "tablets" not in answer.answer_text
    assert "2 years" not in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    assert len(answer.citations) == 1
    assert "18 and 65" in answer.citations[0].statement


def test_age_question_with_mixed_approval_still_hides_non_age_ranges():
    guide = generate_patient_guide(
        _numeric_range_analysis(),
        rewriter=_fake_rewriter,
    )
    age_item = next(i for i in guide.who_can_join.items if "years old" in i.original_text)
    approve_item(age_item)
    # Approve non-age numeric ranges too: they still must not appear for age Qs.
    for item in guide.who_can_join.items:
        if item is not age_item:
            approve_item(item)

    answer = answer_patient_from_guide(
        "Do I need to be 18 years old to join?",
        guide,
    )

    assert "years old" in answer.answer_text
    assert "BMI" not in answer.answer_text
    assert "tablets" not in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text


# ---------------------------------------------------------------------------
# Through the Streamlit app
# ---------------------------------------------------------------------------

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_patient_assistant_hides_unapproved_items():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["pipeline_result"] = {
        "analysis": _analysis(),
        "verification": [],
        "extraction_warnings": [],
        "accuracy_warnings": [],
    }
    with patch("agents.plain_language_agent.ollama_rewriter", side_effect=_fake_rewriter):
        at.session_state["patient_guide"] = _approve_age_only(_guide())
    at.run()

    at.text_input(key="patient_query_input").input("Who can join this study?")
    next(b for b in at.button if b.key == "patient_ask_btn").click()
    at.run()

    assert not at.exception
    answer = at.session_state["patient_history"][-1]["answer"]
    assert "18 and 65" in answer.answer_text
    assert "Type 2 diabetes" not in answer.answer_text
    # The coordinator tabs still show the full protocol; only the patient
    # answer is restricted, so check the rendered answer text itself.
    assert any("In simple words: " + "Participants must be between 18" in m.value for m in at.markdown)
    assert not any(
        "In simple words: " + DIABETES in m.value for m in at.markdown
    )
