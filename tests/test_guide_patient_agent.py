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
    TrialVisit,
)

AGE = "Participants must be between 18 and 65 years old."
DIABETES = "Participants must have a diagnosis of Type 2 diabetes."
EGFR = "eGFR below 45 mL/min/1.73 m2."


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
