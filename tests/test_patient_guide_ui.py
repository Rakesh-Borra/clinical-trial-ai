"""
Tests for the Patient Guide screens.

Runs the real Streamlit app with Streamlit's AppTest, using a cached
pipeline result and a fake rewriter, so no Ollama server is needed.
"""

import threading
from tests.publish_helpers import exact_rows, generate_verified_guide
from tests.publish_helpers import make_publishable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from services import analysis_jobs

from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    SafetyInformation,
    TrialVisit,
)
from ui.patient_guide_tab import item_status, term_definition_answer
from agents.plain_language_agent import explain_statement, generate_patient_guide


def _pipeline_result():
    analysis = ProtocolAnalysis(
        analysis_complete=True,
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
        "analysis_complete": True,
        "verification": exact_rows(analysis),
        "completeness_audit": {"categories": {}},
        "extraction_warnings": [],
        "accuracy_warnings": [],
    }


def _eight_statement_result():
    result = _pipeline_result()
    analysis = result["analysis"]
    analysis.inclusion_criteria.extend(
        EligibilityCriterion(
            description=f"Participants must meet inclusion condition {index}.",
            source_page=1,
        )
        for index in range(2, 4)
    )
    analysis.exclusion_criteria.extend(
        EligibilityCriterion(
            description=f"Participants with exclusion condition {index}.",
            source_page=2,
        )
        for index in range(2, 3)
    )
    analysis.safety_information.append(
        SafetyInformation(
            description="The study team will review safety reports.",
            source_page=4,
        )
    )
    result["verification"] = exact_rows(analysis)
    return result


def _fake_rewriter(original, section, feedback=None):
    # Drops the eGFR numbers so that one item gets flagged.
    if original.startswith("eGFR"):
        return "People with weak kidneys may not be able to join."
    return f"In simple words: {original}"


def _approved_guide(result=None):
    analysis = (result or _pipeline_result())["analysis"]
    guide = generate_verified_guide(analysis, rewriter=_fake_rewriter)
    for item in guide.all_items():
        item.approved = True
        item.rejected = False
    return guide


def _prepare_review_app(result=None, guide=None):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at.session_state["coord_step"] = "3 · Review"
    at.session_state["coord_max_step"] = 2
    at.session_state["pipeline_result"] = result or _pipeline_result()
    at.session_state["coordinator_protocol_id"] = "protocol-eight"
    at.session_state["coordinator_protocols"] = {
        "protocol-eight": {"name": "Eight Statement Study"}
    }
    if guide is not None:
        at.session_state["patient_guide"] = guide
    make_publishable("protocol-eight")
    return at.run()


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _app_with_analysis():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["pipeline_result"] = _pipeline_result()
    return at


def _click(at, key):
    next(b for b in at.button if b.key == key).click()
    return at.run()


def _generated(at):
    """Wait for the background statement generation, then pick up its result."""
    from services import guide_jobs

    assert guide_jobs.jobs.wait_idle(15)
    return at.run()


@pytest.fixture(autouse=True)
def _fresh_job_manager():
    analysis_jobs.jobs.reset()
    yield
    analysis_jobs.jobs.reset()


def _analyze(at, analyzer):
    """Click Analyze with the pipeline mocked, then collect the finished job."""
    with patch("services.protocol_pipeline.analyze_protocol_pdf", analyzer):
        at = _click(at, "analyze_btn")
        job_id = at.session_state["analysis_job_id"]
        assert analysis_jobs.jobs.wait(job_id)
    at = at.run()
    assert at.session_state["pipeline_result"] is not None
    return _click(at, "continue_summary_btn")


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
    assert item_status(item) == "⏳ Waiting for your decision"  # AI warning, no decision
    item.source_status = "verified"
    assert item_status(item) == "⏳ Waiting for your decision"
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
        at.session_state["coord_max_step"] = [
            "1 · Upload",
            "2 · Summary",
            "3 · Review",
        ].index(step)
    return at


def test_app_opens_on_welcome_page():
    at = _app_with_analysis().run()

    assert not at.exception
    html_text = " ".join(str(h.value) for h in at.get("html"))
    assert "Welcome to" in html_text
    assert "Demo navigation only" in html_text
    assert any("#090B0F" in str(item.value) for item in at.get("markdown"))
    assert [widget.key for widget in at.text_input] == ["login_name"]
    assert at.radio(key="login_role").value == "coordinator"
    keys = {b.key for b in at.button}
    assert {"home_btn", "login_continue_btn"} <= keys
    assert len([key for key in keys if key and key.startswith("login_")]) == 1
    assert not at.toggle
    # Nothing from the analysis is shown before sign-in.
    assert not at.tabs


def test_sign_in_opens_each_workspace():
    at = _app_with_analysis().run()
    at.radio(key="login_role").set_value("patient")
    at = at.run()
    at = _click(at, "login_continue_btn")
    assert at.session_state["role"] == "patient"
    assert not at.exception
    assert any("Demo role · Patient/caregiver" in item.value for item in at.caption)
    assert {button.key for button in at.button} >= {"home_btn", "logout_btn"}

    at = _click(at, "logout_btn")
    assert at.session_state["role"] is None

    at.radio(key="login_role").set_value("coordinator")
    at = at.run()
    at = _click(at, "login_continue_btn")
    assert at.session_state["role"] == "coordinator"
    assert not at.exception
    at = _click(at, "home_btn")
    assert at.session_state["role"] is None
    assert not at.exception


def test_coordinator_workspace_has_three_steps():
    at = _as(_app_with_analysis(), "coordinator").run()

    assert not at.exception
    assert at.session_state["coord_step"] == "1 · Upload"
    assert not at.selectbox


def test_navigation_arrows_follow_available_coordinator_steps():
    at = _as(_app_with_analysis(), "coordinator").run()
    assert not at.button(key="nav_previous").disabled
    assert at.button(key="nav_next").disabled

    at = _click(at, "nav_previous")
    assert at.session_state["role"] is None
    assert any("Welcome to" in str(item.value) for item in at.get("html"))
    assert at.button(key="nav_previous").disabled
    assert not at.button(key="nav_next").disabled

    at = _click(at, "nav_next")
    assert at.session_state["role"] == "coordinator"
    assert at.session_state["coord_step"] == "1 · Upload"
    at = _click(at, "continue_summary_btn")
    assert at.session_state["coord_step"] == "2 · Summary"
    at = _click(at, "continue_review_btn")
    assert at.session_state["coord_step"] == "3 · Review"

    at = _click(at, "nav_previous")
    assert at.session_state["coord_step"] == "2 · Summary"
    at = _click(at, "nav_previous")
    assert at.session_state["coord_step"] == "1 · Upload"
    at = _click(at, "nav_previous")
    assert at.session_state["role"] is None
    assert at.button(key="nav_previous").disabled

    at = _click(at, "nav_next")
    assert at.session_state["coord_step"] == "1 · Upload"
    at = _click(at, "nav_next")
    assert at.session_state["coord_step"] == "2 · Summary"
    at = _click(at, "nav_next")
    assert at.session_state["coord_step"] == "3 · Review"
    assert at.button(key="nav_next").disabled


def test_new_navigation_after_back_clears_forward_history():
    at = _as(_app_with_analysis(), "coordinator").run()
    at = _click(at, "continue_summary_btn")
    at = _click(at, "continue_review_btn")
    at = _click(at, "nav_previous")
    assert at.session_state["coord_step"] == "2 · Summary"

    # Visiting the review step again creates a new entry and drops its old
    # forward entry, as a browser does after navigating from a back position.
    at = _click(at, "continue_review_btn")
    history = at.session_state["page_history"]
    index = at.session_state["page_history_index"]
    assert index == len(history) - 1
    assert len(history) == 4
    assert at.session_state["coord_step"] == "3 · Review"
    assert at.button(key="nav_next").disabled


def test_patient_page_history_restores_selection_and_summary():
    at = _app_with_analysis()
    at.session_state["published_protocols"] = {
        "protocol-a": {
            "name": "Diabetes Study",
            "guide": _approved_guide(),
        }
    }
    at = _as(at, "patient").run()
    at.selectbox(key="patient_protocol_select").set_value("protocol-a")
    at = at.run()
    at = _click(at, "view_protocol_summary_btn")
    assert at.session_state["patient_summary_open"]

    at = _click(at, "nav_previous")
    assert at.session_state["role"] == "patient"
    assert not at.session_state["patient_summary_open"]
    assert at.selectbox(key="patient_protocol_select")
    at = _click(at, "nav_previous")
    assert at.session_state["role"] is None

    at = _click(at, "nav_next")
    assert at.session_state["role"] == "patient"
    assert not at.session_state["patient_summary_open"]
    at = _click(at, "nav_next")
    assert at.session_state["patient_summary_open"]
    assert any("Diabetes Study" in str(item.value) for item in at.get("html"))


def test_saving_protocol_name_does_not_clear_cached_analysis():
    at = _app_with_analysis()
    cached_result = at.session_state["pipeline_result"]
    at.session_state["coordinator_protocol_id"] = "protocol-a"
    at.session_state["coordinator_protocols"] = {}
    _as(at, "coordinator")
    at.run()

    at.text_input(key="protocol_name_protocol-a").set_value("Diabetes Study")
    at = _click(at, "save_protocol_name_btn")

    assert not at.exception
    assert at.session_state["coordinator_protocols"]["protocol-a"]["name"] == "Diabetes Study"
    assert at.session_state["pipeline_result"] == cached_result
    assert any("Protocol name saved" in item.value for item in at.success)


def test_protocol_name_cannot_overwrite_another_protocol():
    at = _app_with_analysis()
    at.session_state["coordinator_protocol_id"] = "protocol-b"
    at.session_state["coordinator_protocols"] = {
        "protocol-a": {"name": "Existing Study"}
    }
    _as(at, "coordinator")
    at.run()

    at.text_input(key="protocol_name_protocol-b").set_value(" existing study ")
    at = _click(at, "save_protocol_name_btn")

    assert "protocol-b" not in at.session_state["coordinator_protocols"]
    assert at.session_state["coordinator_protocols"]["protocol-a"]["name"] == "Existing Study"
    assert any("already saved for another protocol" in item.value for item in at.warning)


def test_publication_requires_every_statement_to_be_approved_and_not_rejected(publish_context):
    from app import can_publish_guide, publish_block_reason

    result = _pipeline_result()
    guide = _approved_guide(result)
    assert can_publish_guide(guide, result)
    assert publish_block_reason(guide, result) is None

    guide.all_items()[0].approved = False
    assert publish_block_reason(guide, result) == (
        "Cannot publish: 1 statement(s) still need a coordinator decision. "
        "Approve, edit and approve, or reject each one."
    )
    # A deliberate rejection is a decision: it no longer blocks publication.
    guide.all_items()[0].rejected = True
    assert can_publish_guide(guide, result)
    guide.all_items()[0].rejected = False
    guide.all_items()[0].approved = True
    guide.all_items()[0].analysis_complete = False
    assert not can_publish_guide(guide, result)


def test_coordinator_assistant_is_removed_from_review():
    at = _prepare_review_app()

    assert not at.exception
    assert not any(
        widget.key in {
            "coordinator_query_input",
            "coordinator_ask_btn",
        }
        for widget in [*at.text_input, *at.button]
    )
    visible_text = " ".join(
        [item.value for item in at.caption]
        + [item.value for item in at.markdown]
        + [item.value for item in at.subheader]
        + [item.value for item in at.html]
    )
    assert "Ask about the protocol" not in visible_text
    assert "Ask coordinator assistant" not in visible_text
    assert "study metadata. Answers use verified extracted facts only." not in visible_text


def test_publish_disabled_with_exact_unapproved_statement_reason():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    guide.all_items()[0].approved = False
    at = _prepare_review_app(result, guide)

    assert len(guide.all_items()) == 8
    assert guide.approved_count() == 7
    assert at.button(key="publish_protocol_btn").disabled
    assert any(
        "Cannot publish: 1 statement(s) still need a coordinator decision."
        in warning.value
        for warning in at.warning
    )

    assert at.session_state["published_protocols"] == {}


def test_publish_disabled_with_incomplete_analysis_reason():
    result = _eight_statement_result()
    result["analysis_complete"] = False
    guide = _approved_guide(result)
    at = _prepare_review_app(result, guide)

    assert guide.approved_count() == 8
    assert at.button(key="publish_protocol_btn").disabled
    assert any(
        "Analysis is incomplete" in warning.value
        and "publication and approval remain blocked" in warning.value
        for warning in at.warning
    )
    assert at.session_state["published_protocols"] == {}


def test_publish_eight_approved_statements_and_patient_selection():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    at = _prepare_review_app(result, guide)

    assert len(guide.all_items()) == 8
    assert guide.approved_count() == 8
    assert guide.is_fully_approved()
    assert not at.button(key="publish_protocol_btn").disabled

    at = _click(at, "publish_protocol_btn")
    assert any(
        "Published successfully! Coordinator task completed." in success.value
        for success in at.success
    )
    assert set(at.session_state["published_protocols"]) == {"protocol-eight"}
    published = at.session_state["published_protocols"]["protocol-eight"]
    assert published["name"] == "Eight Statement Study"
    assert len(published["guide"].all_items()) == 8
    assert published["guide"].trial_title == "Eight Statement Study"

    at.session_state["role"] = "patient"
    at = at.run()
    assert at.selectbox(key="patient_protocol_select").options == [
        "Eight Statement Study"
    ]
    assert at.selectbox(key="patient_protocol_select").value is None
    assert not at.chat_input

    at.selectbox(key="patient_protocol_select").set_value("protocol-eight")
    at = at.run()
    assert not at.chat_input
    at = _click(at, "view_protocol_summary_btn")
    assert not at.exception
    assert at.chat_input(key="patient_chat_input")


def test_generate_review_and_publish_flow():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at.session_state["coordinator_protocols"] = {
        "unpublished": {"name": "Unpublished Protocol"},
    }
    at = at.run()
    assert at.session_state["coord_step"] == "1 · Upload"

    at.file_uploader[0].set_value(
        ("protocol.pdf", b"%PDF-1.4 demonstration", "application/pdf")
    )
    at = at.run()
    protocol_id = at.session_state["coordinator_protocol_id"]
    name_key = f"protocol_name_{protocol_id}"
    assert at.text_input(key=name_key)

    at.text_input(key=name_key).set_value("Type 2 Diabetes Treatment Study")
    at = _click(at, "save_protocol_name_btn")
    assert at.session_state["pipeline_result"] is None
    assert at.session_state["coordinator_protocols"][protocol_id]["name"] == (
        "Type 2 Diabetes Treatment Study"
    )

    at = _analyze(at, MagicMock(return_value=_pipeline_result()))
    assert at.session_state["coord_step"] == "2 · Summary"
    with patch(
        "agents.plain_language_agent.ollama_rewriter",
        side_effect=_fake_rewriter,
    ):
        at = _click(at, "continue_review_btn")
        assert at.session_state["coord_step"] == "3 · Review"
        at = _generated(at)

    assert not at.exception
    guide = at.session_state["patient_guide"]
    assert len(guide.all_items()) == 4
    assert len(guide.flagged_items()) == 1  # eGFR item lost its numbers

    # Coordinator approves everything that passed the checks.
    at = _click(at, "guide_approve_all_btn")
    guide = at.session_state["patient_guide"]
    assert guide.approved_count() == 3
    assert not guide.is_fully_approved()
    assert at.button(key="publish_protocol_btn").disabled

    flagged_approval = next(
        button.key
        for button in at.button
        if button.key and button.key.endswith("_approve")
    )
    at = _click(at, flagged_approval)
    make_publishable(at.session_state["coordinator_protocol_id"])
    at = at.run()
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")
    assert any(
        "Published successfully! Coordinator task completed." in item.value
        for item in at.success
    )
    assert protocol_id in at.session_state["published_protocols"]

    # The patient must select the successfully published protocol first.
    at.session_state["role"] = "patient"
    at.run()
    assert at.selectbox(key="patient_protocol_select").value is None
    assert not any("In simple words:" in str(item.value) for item in at.get("html"))
    at.selectbox(key="patient_protocol_select").set_value(protocol_id)
    at = at.run()
    assert at.selectbox(key="patient_protocol_select").options == [
        "Type 2 Diabetes Treatment Study"
    ]
    at = _click(at, "view_protocol_summary_btn")
    patient_text = " ".join(str(h.value) for h in at.get("html"))
    assert not at.exception
    assert "In simple words: Participants must be between 18 and 65 years old." in patient_text
    assert "weak kidneys" not in patient_text
    assert "T2D-001" not in patient_text
    from app import cleanup_temp_pdf

    cleanup_temp_pdf(at.session_state.get("temp_pdf_path"))


def test_coordinator_can_reject_a_statement():
    with patch(
        "agents.plain_language_agent.ollama_rewriter",
        side_effect=_fake_rewriter,
    ):
        at = _as(_app_with_analysis(), "coordinator", "3 · Review").run()
        at = _generated(at)

    reject_key = next(b.key for b in at.button if b.key and b.key.endswith("_reject"))
    at = _click(at, reject_key)
    guide = at.session_state["patient_guide"]
    assert guide.rejected_count() == 1
    assert not any(i.approved and i.rejected for i in guide.all_items())


def test_patient_empty_state_before_publish():
    at = _app_with_analysis()
    at.session_state["patient_guide"] = _approved_guide()
    at.session_state["coordinator_protocols"] = {
        "unpublished": {"name": "Unpublished Protocol"}
    }
    at = _as(at, "patient").run()

    assert not at.exception
    assert any(
        "No published clinical trial protocols are available yet." in item.value
        for item in at.info
    )
    assert not at.selectbox


def test_patient_qna_uses_selected_protocol():
    at = _app_with_analysis()
    at.session_state["published_protocols"] = {
        "protocol-a": {
            "name": "Diabetes Study",
            "guide": _approved_guide(),
        }
    }
    at = _as(at, "patient").run()
    at.selectbox(key="patient_protocol_select").set_value("protocol-a")
    at = at.run()
    at = _click(at, "view_protocol_summary_btn")
    at.chat_input(key="patient_chat_input").set_value("What visits are there?").run()

    assert not at.exception
    history = at.session_state["patient_history"]
    assert history[-1]["answer"].intent == "visits"
    assert "Follow-up Visit" in history[-1]["answer"].answer_text


def test_switching_protocols_clears_summary_and_chat_context():
    guide_a = _approved_guide()
    guide_b = _approved_guide()
    guide_b.trial_title = "Blood Pressure Study"
    guide_b.visits.items[0].plain_text = "In simple words: Visit B is on Day 90."
    at = _app_with_analysis()
    at.session_state["published_protocols"] = {
        "protocol-a": {"name": "Diabetes Study", "guide": guide_a},
        "protocol-b": {"name": "Blood Pressure Study", "guide": guide_b},
    }
    at = _as(at, "patient").run()
    at.selectbox(key="patient_protocol_select").set_value("protocol-a")
    at = at.run()
    at = _click(at, "view_protocol_summary_btn")
    at.chat_input(key="patient_chat_input").set_value("What visits are there?").run()
    assert at.session_state["patient_history"]

    at.selectbox(key="patient_protocol_select").set_value("protocol-b")
    at = at.run()
    assert at.session_state["patient_history"] == []
    assert not any("Visit B is on Day 90" in str(item.value) for item in at.get("html"))
    at = _click(at, "view_protocol_summary_btn")
    patient_text = " ".join(str(item.value) for item in at.get("html"))
    assert "Blood Pressure Study" in patient_text
    assert "In simple words: Visit B is on Day 90." in patient_text
    assert "Type 2 Diabetes Treatment Study" not in patient_text


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
        at = _as(_app_with_analysis(), "coordinator", "3 · Review")
        at.session_state["pipeline_result"] = result
        at = _generated(at.run())

    assert not at.exception
    guide = at.session_state["patient_guide"]
    assert guide.analysis_complete is False
    # Approval is per statement now; only publication needs a complete analysis.
    first_item_approval = next(
        b for b in at.button if b.key and b.key.endswith("_approve")
    )
    assert not first_item_approval.disabled
    assert at.button(key="publish_protocol_btn").disabled

    # Even pre-approved state cannot publish incomplete content to patients.
    guide.all_items()[0].approved = True
    at.session_state["role"] = "patient"
    at.run()
    assert not at.exception
    assert any(
        "No published clinical trial protocols are available yet." in item.value
        for item in at.info
    )
    patient_text = " ".join(str(item.value) for item in at.get("html"))
    assert "In simple words: Participants must be between 18 and 65 years old." not in patient_text


def test_provider_messaging_uses_safe_configured_label(monkeypatch):
    monkeypatch.setenv("CTAI_MODEL_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    release = threading.Event()

    def _slow_rewriter(original, section, feedback=None):
        release.wait(10)
        return f"In simple words: {original}"

    with patch("agents.plain_language_agent.ollama_rewriter", _slow_rewriter):
        at = _as(_app_with_analysis(), "coordinator", "3 · Review").run()
        try:
            assert not at.exception
            info = " ".join(element.value for element in at.info)
            assert "OpenRouter (cloud)" in info
            assert "secret-test-key" not in info
        finally:
            release.set()
            _generated(at)


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


# ---------------------------------------------------------------------------
# Progress is kept across navigation, Home, logout, and uploader recreation
# ---------------------------------------------------------------------------

def _upload(at, content=b"%PDF-1.4 protocol A", name="protocol.pdf"):
    at.file_uploader[0].set_value((name, content, "application/pdf"))
    return at.run()


def _reviewed_session(approve=True):
    """Upload, name, analyze, generate, and (optionally) approve the guide."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at = at.run()
    at = _upload(at)
    protocol_id = at.session_state["coordinator_protocol_id"]
    at.text_input(key=f"protocol_name_{protocol_id}").set_value("Persistent Study")
    at = _click(at, "save_protocol_name_btn")
    analyzer = MagicMock(return_value=_eight_statement_result())
    at = _analyze(at, analyzer)
    with patch("agents.plain_language_agent.ollama_rewriter", side_effect=_fake_rewriter):
        at = _click(at, "continue_review_btn")
        at = _generated(at)
    if approve:
        at = _click(at, "guide_approve_all_btn")
        for button in [b for b in at.button if b.key and b.key.endswith("_approve")]:
            at = _click(at, button.key)
        make_publishable(protocol_id)
        at = at.run()
    return at, analyzer, protocol_id


def _back_to_home(at):
    for _ in range(10):
        if at.session_state["role"] is None:
            return at
        at = _click(at, "nav_previous")
    raise AssertionError("never reached Home")


def _forward_to_end(at):
    while not at.button(key="nav_next").disabled:
        at = _click(at, "nav_next")
    return at


def _progress(at):
    guide = at.session_state["patient_guide"]
    return {
        "pdf": at.session_state["uploaded_name"],
        "fingerprint": at.session_state["uploaded_fingerprint"],
        "protocol_id": at.session_state["coordinator_protocol_id"],
        "analysis": id(at.session_state["pipeline_result"]),
        "approved": [item.approved for item in guide.all_items()],
        "rejected": [item.rejected for item in guide.all_items()],
        "name": at.session_state["coordinator_protocols"][
            at.session_state["coordinator_protocol_id"]
        ]["name"],
    }


def test_back_and_forward_preserve_upload_analysis_and_all_approvals():
    at, analyzer, _ = _reviewed_session()
    before = _progress(at)
    assert len(before["approved"]) == 8 and all(before["approved"])
    assert at.session_state["coord_step"] == "3 · Review"

    at = _back_to_home(at)
    assert not at.exception
    assert _progress(at) == before

    at = _forward_to_end(at)
    assert not at.exception
    assert at.session_state["role"] == "coordinator"
    assert at.session_state["coord_step"] == "3 · Review"
    assert _progress(at) == before
    assert not at.button(key="publish_protocol_btn").disabled
    assert analyzer.call_count == 1


def test_upload_step_restores_document_and_name_after_navigation():
    at, analyzer, protocol_id = _reviewed_session()
    at = _click(at, "nav_previous")
    at = _click(at, "nav_previous")
    assert at.session_state["coord_step"] == "1 · Upload"

    assert at.text_input(key=f"protocol_name_{protocol_id}").value == "Persistent Study"
    assert any("Analysis cached for" in item.value for item in at.success)
    assert not at.button(key="analyze_btn") or at.button(key="analyze_btn").disabled
    assert at.button(key="continue_summary_btn")
    assert analyzer.call_count == 1
    assert not at.exception


def test_returning_home_and_reopening_coordinator_keeps_progress():
    at, analyzer, _ = _reviewed_session()
    before = _progress(at)

    at = _click(at, "home_btn")
    assert at.session_state["role"] is None
    at.radio(key="login_role").set_value("coordinator")
    at = at.run()
    at = _click(at, "login_continue_btn")

    assert at.session_state["role"] == "coordinator"
    assert _progress(at) == before
    at = _click(at, "continue_summary_btn")
    at = _click(at, "continue_review_btn")
    assert _progress(at) == before
    assert not at.button(key="publish_protocol_btn").disabled
    assert analyzer.call_count == 1


def test_logout_keeps_coordinator_progress_but_not_patient_access():
    at, _, protocol_id = _reviewed_session()
    before = _progress(at)

    at = _click(at, "logout_btn")
    assert at.session_state["role"] is None
    assert _progress(at) == before
    assert at.session_state["published_protocols"] == {}

    # Role separation: a patient never sees unpublished coordinator work.
    at.radio(key="login_role").set_value("patient")
    at = at.run()
    at = _click(at, "login_continue_btn")
    assert at.session_state["role"] == "patient"
    assert any(
        "No published clinical trial protocols are available yet." in item.value
        for item in at.info
    )
    assert not at.selectbox
    assert "Persistent Study" not in " ".join(str(h.value) for h in at.get("html"))
    assert _progress(at) == before


def test_publication_works_after_navigation_and_survives_patient_visit():
    at, _, protocol_id = _reviewed_session()
    at = _back_to_home(at)
    at = _forward_to_end(at)

    at = _click(at, "publish_protocol_btn")
    assert any(
        "Published successfully! Coordinator task completed." in item.value
        for item in at.success
    )
    published = at.session_state["published_protocols"]
    assert published[protocol_id]["name"] == "Persistent Study"

    at = at.run()
    at = _click(at, "preview_patient_btn")
    assert at.session_state["role"] == "patient"
    at = _click(at, "nav_previous")
    assert at.session_state["role"] == "coordinator"
    assert protocol_id in at.session_state["published_protocols"]
    assert _progress(at)["pdf"] == "protocol.pdf"


def test_navigation_does_not_bypass_unapproved_publication_guardrail():
    at, _, _ = _reviewed_session(approve=False)
    assert at.button(key="publish_protocol_btn").disabled

    at = _back_to_home(at)
    at = _forward_to_end(at)

    assert at.button(key="publish_protocol_btn").disabled
    assert at.session_state["published_protocols"] == {}
    assert not any(item.approved for item in at.session_state["patient_guide"].all_items())


def test_unanalyzed_upload_survives_navigation_without_reanalysis():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at = _upload(at.run())
    pdf_path = at.session_state["temp_pdf_path"]
    fingerprint = at.session_state["uploaded_fingerprint"]

    analyzer = MagicMock(return_value=_pipeline_result())
    with patch("services.protocol_pipeline.analyze_protocol_pdf", analyzer):
        at = _click(at, "nav_previous")
        assert at.session_state["role"] is None
        at = _click(at, "nav_next")

    assert at.session_state["temp_pdf_path"] == pdf_path
    assert at.session_state["uploaded_fingerprint"] == fingerprint
    assert any("Ready to analyze" in item.value for item in at.success)
    assert not at.button(key="analyze_btn").disabled
    assert analyzer.call_count == 0
    from app import cleanup_temp_pdf

    cleanup_temp_pdf(pdf_path)


def test_different_pdf_requires_confirmation_and_never_reuses_old_analysis():
    at, analyzer, old_id = _reviewed_session()
    at = _click(at, "nav_previous")
    at = _click(at, "nav_previous")
    old = _progress(at)

    # Same name and size as the original upload, different content.
    original = b"%PDF-1.4 protocol A"
    other = b"%PDF-1.4 protocol B"
    assert len(original) == len(other)
    at = _upload(at, other)

    assert any("different from the current protocol" in w.value for w in at.warning)
    assert _progress(at) == old
    assert at.session_state["coordinator_protocol_id"] == old_id

    at = _click(at, "replace_protocol_btn")
    assert at.session_state["coordinator_protocol_id"] != old_id
    assert at.session_state["pipeline_result"] is None
    assert "patient_guide" not in at.session_state
    assert any("Ready to analyze" in item.value for item in at.success)
    assert not at.button(key="analyze_btn").disabled
    assert analyzer.call_count == 1
    from app import cleanup_temp_pdf

    cleanup_temp_pdf(at.session_state["temp_pdf_path"])


def test_keep_current_protocol_discards_the_rejected_replacement():
    at, _, old_id = _reviewed_session()
    at = _click(at, "nav_previous")
    at = _click(at, "nav_previous")
    old = _progress(at)

    at = _upload(at, b"%PDF-1.4 protocol Z")
    at = _click(at, "keep_protocol_btn")

    assert _progress(at) == old
    assert at.session_state["coordinator_protocol_id"] == old_id
    assert not any("different from the current protocol" in w.value for w in at.warning)


def test_upload_fingerprint_depends_on_content():
    from app import upload_fingerprint
    from types import SimpleNamespace

    def uploaded(content):
        return SimpleNamespace(
            name="same.pdf",
            size=len(content),
            getbuffer=lambda: memoryview(content),
        )

    assert upload_fingerprint(uploaded(b"%PDF-1.4 A")) != upload_fingerprint(uploaded(b"%PDF-1.4 B"))
    assert upload_fingerprint(uploaded(b"%PDF-1.4 A")) == upload_fingerprint(uploaded(b"%PDF-1.4 A"))


def test_rerun_does_not_trigger_reanalysis():
    at, analyzer, _ = _reviewed_session()
    for _ in range(3):
        at = at.run()
    assert analyzer.call_count == 1
    assert at.session_state["pipeline_result"] is not None


# ---------------------------------------------------------------------------
# Background analysis jobs survive navigation and reruns
# ---------------------------------------------------------------------------

class _GatedAnalyzer:
    """Pipeline stand-in that blocks until released, so tests can observe it running."""

    def __init__(self, result=None, error=None):
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls = 0
        self._result = result if result is not None else _pipeline_result()
        self._error = error

    def __call__(self, path):
        self.calls += 1
        self.started.set()
        assert self.release.wait(10), "analyzer was never released"
        if self._error:
            raise self._error
        return self._result


def _start_gated(analyzer, content=b"%PDF-1.4 protocol A", at=None):
    at = at or AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at = _upload(at.run(), content)
    protocol_id = at.session_state["coordinator_protocol_id"]
    at.text_input(key=f"protocol_name_{protocol_id}").set_value("Slow Study")
    at = _click(at, "save_protocol_name_btn")
    patcher = patch("services.protocol_pipeline.analyze_protocol_pdf", analyzer)
    patcher.start()
    at = _click(at, "analyze_btn")
    assert analyzer.started.wait(5)
    return at, patcher


def _finish(at, analyzer, patcher):
    analyzer.release.set()
    assert analysis_jobs.jobs.wait(at.session_state["analysis_job_id"])
    patcher.stop()
    at.session_state["_analysis_watched"] = True
    return at.run()


def test_analysis_keeps_running_after_navigating_home_and_back():
    analyzer = _GatedAnalyzer()
    at, patcher = _start_gated(analyzer)
    try:
        job_id = at.session_state["analysis_job_id"]
        assert analysis_jobs.jobs.get(job_id).status == analysis_jobs.RUNNING
        assert at.button(key="analyze_btn").disabled
        assert any("keeps running if you leave this page" in i.value for i in at.info)

        at = _click(at, "nav_previous")
        assert at.session_state["role"] is None
        assert analysis_jobs.jobs.get(job_id).status == analysis_jobs.RUNNING

        at = _click(at, "nav_next")
        assert at.session_state["role"] == "coordinator"
        assert at.session_state["analysis_job_id"] == job_id
        assert at.session_state["pipeline_result"] is None
        assert at.button(key="analyze_btn").disabled
        assert any("keeps running if you leave this page" in i.value for i in at.info)

        at = _finish(at, analyzer, patcher)
    finally:
        analyzer.release.set()
        patcher.stop()

    assert analyzer.calls == 1
    assert at.session_state["pipeline_result"] is not None
    assert at.session_state["analysis_job_id"] is None
    assert at.session_state["coord_step"] == "2 · Summary"


def test_result_finishing_while_away_is_restored_without_rerunning():
    analyzer = _GatedAnalyzer()
    at, patcher = _start_gated(analyzer)
    try:
        at = _click(at, "nav_previous")
        assert at.session_state["role"] is None
        analyzer.release.set()
        assert analysis_jobs.jobs.wait(at.session_state["analysis_job_id"])
    finally:
        patcher.stop()

    at = _click(at, "nav_next")
    assert at.session_state["role"] == "coordinator"
    assert at.session_state["pipeline_result"] is not None
    assert at.session_state["analysis_job_id"] is None
    assert at.session_state["coord_step"] == "1 · Upload"
    assert any("Analysis cached for" in i.value for i in at.success)
    assert at.button(key="continue_summary_btn")
    assert analyzer.calls == 1


def test_reruns_and_repeated_analyze_do_not_create_duplicate_jobs():
    analyzer = _GatedAnalyzer()
    at, patcher = _start_gated(analyzer)
    try:
        job_id = at.session_state["analysis_job_id"]
        for _ in range(3):
            at = at.run()
        at = _click(at, "nav_previous")
        at = _click(at, "nav_next")
        assert at.session_state["analysis_job_id"] == job_id
        assert len(analysis_jobs.jobs._jobs) == 1

        assert at.button(key="analyze_btn").disabled
        at = _finish(at, analyzer, patcher)
    finally:
        analyzer.release.set()
        patcher.stop()
    assert analyzer.calls == 1


def test_manager_deduplicates_active_jobs_for_same_pdf():
    release = threading.Event()
    calls = []

    def runner(path):
        calls.append(path)
        release.wait(5)
        return "result"

    first = analysis_jobs.jobs.submit(runner, "/a.pdf", "p1", "fp1")
    second = analysis_jobs.jobs.submit(runner, "/a.pdf", "p1", "fp1")
    assert first == second
    other = analysis_jobs.jobs.submit(runner, "/b.pdf", "p2", "fp2")
    assert other != first
    release.set()
    assert analysis_jobs.jobs.wait(first) and analysis_jobs.jobs.wait(other)
    assert analysis_jobs.jobs.get(first).status == analysis_jobs.COMPLETED
    assert analysis_jobs.jobs.get(first).result == "result"
    assert len(calls) == 2


def test_failed_job_shows_error_and_keeps_upload_and_name():
    analyzer = _GatedAnalyzer(error=ValueError("The protocol PDF contains no extractable text"))
    at, patcher = _start_gated(analyzer)
    pdf_path = at.session_state["temp_pdf_path"]
    try:
        at = _finish(at, analyzer, patcher)
    finally:
        analyzer.release.set()
        patcher.stop()

    assert not at.exception
    assert at.session_state["pipeline_result"] is None
    assert any("Could not analyze this PDF" in e.value for e in at.error)
    assert at.session_state["temp_pdf_path"] == pdf_path
    assert at.session_state["uploaded_name"] == "protocol.pdf"
    protocol_id = at.session_state["coordinator_protocol_id"]
    assert at.session_state["coordinator_protocols"][protocol_id]["name"] == "Slow Study"
    assert not at.button(key="analyze_btn").disabled


def test_replacing_pdf_discards_running_job_and_blocks_stale_result():
    analyzer = _GatedAnalyzer(result=_eight_statement_result())
    at, patcher = _start_gated(analyzer)
    try:
        old_id = at.session_state["coordinator_protocol_id"]
        old_job = at.session_state["analysis_job_id"]

        at = _upload(at, b"%PDF-1.4 protocol B")
        assert any("different from the current protocol" in w.value for w in at.warning)
        assert at.session_state["analysis_job_id"] == old_job
        at = _click(at, "replace_protocol_btn")

        new_id = at.session_state["coordinator_protocol_id"]
        assert new_id != old_id
        assert at.session_state["analysis_job_id"] is None
        assert analysis_jobs.jobs.get(old_job).status == analysis_jobs.DISCARDED

        analyzer.release.set()
        assert analysis_jobs.jobs.wait(old_job)
    finally:
        analyzer.release.set()
        patcher.stop()

    at = at.run()
    assert at.session_state["pipeline_result"] is None
    assert analysis_jobs.jobs.get(old_job).result is None
    assert any("Ready to analyze" in i.value for i in at.success)


def test_stale_job_for_other_protocol_is_never_attached():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at = _upload(at.run())
    stale = analysis_jobs.jobs.submit(
        lambda path: _pipeline_result(), "someone-else", "other-fingerprint", "x"
    )
    assert analysis_jobs.jobs.wait(stale)
    at.session_state["analysis_job_id"] = stale
    at = at.run()

    assert at.session_state["pipeline_result"] is None
    assert at.session_state["analysis_job_id"] is None
    assert analysis_jobs.jobs.get(stale) is None


def test_publishing_is_blocked_while_analysis_is_running():
    from app import publish_block_reason

    analyzer = _GatedAnalyzer()
    at, patcher = _start_gated(analyzer)
    try:
        at.session_state["_coord_goto"] = "3 · Review"
        at = at.run()
        assert at.session_state["coord_step"] == "3 · Review"
        assert at.session_state["published_protocols"] == {}
        assert any("keeps running if you leave this page" in i.value for i in at.info)
        assert "publish_protocol_btn" not in {b.key for b in at.button}
        at = _finish(at, analyzer, patcher)
    finally:
        analyzer.release.set()
        patcher.stop()

    with patch("app.active_analysis_job", return_value=object()):
        assert "still running" in publish_block_reason(None, None)

    # A completed-but-incomplete analysis also stays blocked.
    incomplete = _pipeline_result()
    incomplete["analysis_complete"] = False
    assert "incomplete" in publish_block_reason(_approved_guide(incomplete), incomplete)


def test_logout_during_analysis_keeps_job_and_result_for_same_session():
    analyzer = _GatedAnalyzer()
    at, patcher = _start_gated(analyzer)
    try:
        job_id = at.session_state["analysis_job_id"]
        at = _click(at, "logout_btn")
        assert at.session_state["role"] is None
        assert at.session_state["analysis_job_id"] == job_id
        analyzer.release.set()
        assert analysis_jobs.jobs.wait(job_id)
    finally:
        patcher.stop()

    at.radio(key="login_role").set_value("patient")
    at = at.run()
    at = _click(at, "login_continue_btn")
    # Patients never see in-progress or unpublished analysis.
    assert at.session_state["pipeline_result"] is not None
    assert not at.selectbox
    assert any(
        "No published clinical trial protocols are available yet." in i.value
        for i in at.info
    )
