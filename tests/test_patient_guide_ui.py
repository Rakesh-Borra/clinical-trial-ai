"""
Tests for the Patient Guide screens.

Runs the real Streamlit app with Streamlit's AppTest, using a cached
pipeline result and a fake rewriter, so no Ollama server is needed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialVisit,
)
from ui.patient_guide_tab import item_status, term_definition_answer
from agents.plain_language_agent import explain_statement


def _pipeline_result():
    analysis = ProtocolAnalysis(
        trial_title="Type 2 Diabetes Treatment Study",
        protocol_id="T2D-001",
        inclusion_criteria=[
            EligibilityCriterion(
                description="Participants must be between 18 and 65 years old.",
                source_page=1,
            ),
        ],
        exclusion_criteria=[
            EligibilityCriterion(
                description="eGFR below 45 mL/min/1.73 m2.",
                source_page=1,
            ),
        ],
        visits=[TrialVisit(visit_name="Follow-up Visit", timing="Day 30", source_page=2)],
        safety_information=[
            SafetyInformation(
                description="Participants will be monitored for adverse events.",
                source_page=2,
            )
        ],
    )
    return {
        "analysis": analysis,
        "verification": [],
        "extraction_warnings": [],
        "accuracy_warnings": [],
    }


def _fake_rewriter(original, section, feedback=None):
    # Drops the eGFR numbers so that one item gets flagged.
    if original.startswith("eGFR"):
        return "People with weak kidneys may not be able to join."
    return f"In simple words: {original}"


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _app_with_analysis():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["pipeline_result"] = _pipeline_result()
    return at


def _click(at, key):
    next(b for b in at.button if b.key == key).click()
    return at.run()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_term_definition_answer():
    answer = term_definition_answer("What does HbA1c mean?")

    assert answer.intent == "term_definition"
    assert "HbA1c" in answer.answer_text
    assert "average blood sugar" in answer.answer_text
    assert answer.disclaimers


def test_term_definition_answer_ignores_other_questions():
    assert term_definition_answer("Who can join this study?") is None


def test_item_status_labels():
    item = explain_statement("Age 18.", "inclusion_criteria", rewriter=lambda o, s, f=None: o)
    assert item_status(item) == "⏳ Waiting for approval"
    item.approved = True
    assert item_status(item) == "✅ Approved"


# ---------------------------------------------------------------------------
# End-to-end through the Streamlit app
# ---------------------------------------------------------------------------

def test_app_shows_four_tabs():
    at = _app_with_analysis().run()

    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Protocol Summary",
        "Coordinator Assistant",
        "Patient Guide Review",
        "Patient Assistant",
    ]


def test_generate_review_and_publish_flow():
    with patch(
        "agents.plain_language_agent.ollama_rewriter",
        side_effect=_fake_rewriter,
    ):
        at = _app_with_analysis().run()
        at = _click(at, "guide_generate_btn")

    assert not at.exception
    guide = at.session_state["patient_guide"]
    assert len(guide.all_items()) == 4
    assert len(guide.flagged_items()) == 1  # eGFR item lost its numbers

    # Before approval, patients see nothing from the guide.
    patient_text = " ".join(m.value for m in at.markdown)
    assert "In simple words" not in patient_text

    # Coordinator approves everything that passed the checks.
    at = _click(at, "guide_approve_all_btn")
    guide = at.session_state["patient_guide"]
    assert guide.approved_count() == 3
    assert not guide.is_fully_approved()

    patient_text = " ".join(m.value for m in at.markdown)
    assert "In simple words: Participants must be between 18 and 65 years old." in patient_text
    # The flagged eGFR item stays hidden from patients.
    assert "weak kidneys" not in patient_text


def test_patient_can_ask_what_a_word_means():
    at = _app_with_analysis().run()
    at.text_input(key="patient_query_input").input("What does eGFR mean?")
    at = _click(at, "patient_ask_btn")

    assert not at.exception
    history = at.session_state["patient_history"]
    assert history[-1]["answer"].intent == "term_definition"
    assert "kidneys" in history[-1]["answer"].answer_text


def test_clear_analysis_removes_guide():
    import app

    class _State(dict):
        def __getattr__(self, key):
            return self.get(key)

        def __setattr__(self, key, value):
            self[key] = value

    state = _State(patient_guide=object(), temp_pdf_path=None)
    mock_st = MagicMock()
    mock_st.session_state = state

    with patch.object(app, "st", mock_st), patch(
        "ui.patient_guide_tab.st", mock_st
    ), patch.object(app, "cleanup_temp_pdf"):
        app.clear_analysis_state(keep_upload=True)

    assert "patient_guide" not in state
