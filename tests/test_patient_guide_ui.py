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

def _as(at, role, step=None):
    at.session_state["role"] = role
    at.session_state["user_name"] = ""
    if step:
        at.session_state["coord_step"] = step
    return at


def test_app_opens_on_welcome_page():
    at = _app_with_analysis().run()

    assert not at.exception
    html_text = " ".join(str(h.value) for h in at.get("html"))
    assert "Welcome to" in html_text
    assert any(
        "Demo role navigation only" in element.value for element in at.info
    )
    keys = {b.key for b in at.button}
    assert {"login_coordinator_btn", "login_patient_btn"} <= keys
    # Nothing from the analysis is shown before sign-in.
    assert not at.tabs


def test_sign_in_opens_each_workspace():
    at = _app_with_analysis().run()
    at = _click(at, "login_patient_btn")
    assert at.session_state["role"] == "patient"
    assert not at.exception

    at = _click(at, "switch_role_btn")
    assert at.session_state["role"] is None

    at = _click(at, "login_coordinator_btn")
    assert at.session_state["role"] == "coordinator"
    assert not at.exception


def test_coordinator_workspace_has_four_steps():
    at = _as(_app_with_analysis(), "coordinator").run()

    assert not at.exception
    assert at.session_state["coord_step"] == "1 · Upload"
    assert at.selectbox(key="doc_type").value == "Clinical trial protocol"


def test_generate_review_and_publish_flow():
    with patch(
        "agents.plain_language_agent.ollama_rewriter",
        side_effect=_fake_rewriter,
    ):
        at = _as(_app_with_analysis(), "coordinator", "3 · Review").run()
        at = _click(at, "guide_generate_btn")

    assert not at.exception
    guide = at.session_state["patient_guide"]
    assert len(guide.all_items()) == 4
    assert len(guide.flagged_items()) == 1  # eGFR item lost its numbers

    # Coordinator approves everything that passed the checks.
    at = _click(at, "guide_approve_all_btn")
    guide = at.session_state["patient_guide"]
    assert guide.approved_count() == 3
    assert not guide.is_fully_approved()

    # The patient workspace shows approved items only.
    at.session_state["role"] = "patient"
    at.run()
    patient_text = " ".join(str(h.value) for h in at.get("html"))
    assert not at.exception
    assert "In simple words: Participants must be between 18 and 65 years old." in patient_text
    assert "weak kidneys" not in patient_text


def test_coordinator_can_reject_a_statement():
    with patch(
        "agents.plain_language_agent.ollama_rewriter",
        side_effect=_fake_rewriter,
    ):
        at = _as(_app_with_analysis(), "coordinator", "3 · Review").run()
        at = _click(at, "guide_generate_btn")

    reject_key = next(b.key for b in at.button if b.key and b.key.endswith("_reject"))
    at = _click(at, reject_key)
    guide = at.session_state["patient_guide"]
    assert guide.rejected_count() == 1
    assert not any(i.approved and i.rejected for i in guide.all_items())


def test_patient_empty_state_before_publish():
    at = _as(_app_with_analysis(), "patient").run()

    assert not at.exception
    html_text = " ".join(str(h.value) for h in at.get("html"))
    assert "study guide is on its way" in html_text


def test_patient_can_ask_what_a_word_means():
    at = _as(_app_with_analysis(), "patient").run()
    at.chat_input(key="patient_chat_input").set_value("What does eGFR mean?").run()

    assert not at.exception
    history = at.session_state["patient_history"]
    assert history[-1]["answer"].intent == "term_definition"
    assert "kidneys" in history[-1]["answer"].answer_text


def test_incomplete_analysis_is_shown_on_coordinator_summary():
    result = _pipeline_result()
    result["analysis_complete"] = False
    result["incomplete_reasons"] = ["Chunk 2 failed extraction."]
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["pipeline_result"] = result
    _as(at, "coordinator", "2 · Summary")
    at.run()

    assert not at.exception
    assert any("Analysis is incomplete" in warning.value for warning in at.warning)
    assert any("Chunk 2 failed extraction." in warning.value for warning in at.warning)


def test_incomplete_guide_disables_approval_and_blocks_patient_view():
    result = _pipeline_result()
    result["analysis_complete"] = False
    result["analysis"].analysis_complete = False
    result["incomplete_reasons"] = ["A protocol chunk failed."]
    with patch(
        "agents.plain_language_agent.ollama_rewriter",
        side_effect=_fake_rewriter,
    ):
        at = _as(_app_with_analysis(), "coordinator", "3 · Review").run()
        at.session_state["pipeline_result"] = result
        at = _click(at, "guide_generate_btn")

    assert not at.exception
    guide = at.session_state["patient_guide"]
    assert guide.analysis_complete is False
    approve_all = next(b for b in at.button if b.key == "guide_approve_all_btn")
    assert approve_all.disabled
    first_item_approval = next(
        b for b in at.button if b.key and b.key.endswith("_approve")
    )
    assert first_item_approval.disabled

    # Even pre-approved state cannot publish incomplete content to patients.
    guide.all_items()[0].approved = True
    at.session_state["role"] = "patient"
    at.run()
    assert not at.exception
    assert any(
        "protocol analysis is incomplete" in warning.value
        for warning in at.warning
    )
    patient_text = " ".join(str(h.value) for h in at.get("html"))
    assert "In simple words: Participants must be between 18 and 65 years old." not in patient_text


def test_provider_messaging_uses_safe_configured_label(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    at = _as(_app_with_analysis(), "coordinator", "3 · Review").run()

    assert not at.exception
    info = " ".join(element.value for element in at.info)
    assert "OpenRouter (cloud)" in info
    assert "secret-test-key" not in info


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


def test_summary_separates_routine_corrections_from_items_needing_review():
    result = _pipeline_result()
    result["accuracy_warnings"] = [
        "procedures: corrected source_page for 'Fasting blood draw' from 6 → 8.",
        "procedures: unsupported statement 'Breakfast provision' was not found on any source page.",
    ]
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["pipeline_result"] = result
    _as(at, "coordinator", "2 · Summary")
    at.run()

    assert not at.exception
    warning_text = " ".join(w.value for w in at.warning)
    assert "Breakfast provision" in warning_text
    assert "corrected source_page" not in warning_text
    assert any("automatic correction" in e.label for e in at.expander)
