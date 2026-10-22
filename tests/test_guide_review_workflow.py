"""Automatic statement generation, per-statement decisions, and filters."""

import threading
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from agents.guide_patient_agent import answer_patient_from_guide
from agents.plain_language_agent import approve_item, edit_item, reject_item
from services import guide_jobs
from services.guide_review import (
    WITHHELD_MESSAGE,
    build_published_snapshot,
    decision_block_reason,
    patient_visible_guide,
    review_state,
    state_counts,
    with_coordinator_referral,
)
from tests.test_patient_guide_ui import (
    APP_PATH,
    _app_with_analysis,
    _approved_guide,
    _as,
    _click,
    _eight_statement_result,
    _generated,
    _pipeline_result,
    _prepare_review_app,
)
from tests.test_patient_portal_layout import _patient_app
from app import publish_block_reason


def _item(guide, text):
    return next(i for i in guide.all_items() if text in i.original_text)


def _ask(at, text):
    at.chat_input(key="patient_chat_input").set_value(text)
    return at.run()


def _flags(at, key):
    return [b for b in at.button if b.key == key]


# --- Task 1: statements are generated when the Review page opens ---------


def test_review_statements_navigates_and_generates_missing_statements():
    at = _as(_app_with_analysis(), "coordinator", "2 · Summary").run()
    assert guide_jobs.jobs.started == 0
    at = _click(at, "continue_review_btn")
    assert at.session_state["coord_step"] == "3 · Review"
    assert guide_jobs.jobs.started == 1
    at = _generated(at)
    guide = at.session_state["patient_guide"]
    assert len(guide.all_items()) == 4
    assert not at.exception
    # Every statement starts Pending: nothing is approved automatically.
    assert state_counts(guide)["pending"] == 4


def test_progress_is_shown_while_generating_and_no_duplicates_start():
    release = threading.Event()

    def slow(original, section, feedback=None):
        release.wait(10)
        return f"In simple words: {original}"

    with patch("agents.plain_language_agent.ollama_rewriter", slow):
        at = _prepare_review_app()
        try:
            assert at.session_state["patient_guide"] is None if "patient_guide" in at.session_state else True
            assert any("keeps running" in i.value for i in at.info)
            assert not _flags(at, "publish_protocol_btn")
            for _ in range(3):
                at = at.run()
            assert guide_jobs.jobs.started == 1
        finally:
            release.set()
            at = _generated(at)
    assert len(at.session_state["patient_guide"].all_items()) == 4
    assert guide_jobs.jobs.started == 1


def test_existing_statements_are_reused_without_regenerating():
    guide = _approved_guide()
    at = _prepare_review_app(guide=guide)
    at = at.run()
    assert guide_jobs.jobs.started == 0
    assert at.session_state["patient_guide"] is guide


def test_generated_statements_survive_leaving_and_returning():
    at = _as(_app_with_analysis(), "coordinator", "2 · Summary").run()
    at = _generated(_click(at, "continue_review_btn"))
    guide = at.session_state["patient_guide"]
    at = _click(at, "nav_previous")
    assert at.session_state["coord_step"] == "2 · Summary"
    at = _click(at, "continue_review_btn")
    assert at.session_state["patient_guide"] is guide
    assert guide_jobs.jobs.started == 1


def test_generation_failure_shows_error_and_retry_works_once():
    with patch(
        "ui.patient_guide_tab.generate_patient_guide",
        side_effect=RuntimeError("model exploded"),
    ):
        at = _prepare_review_app()
        at = _generated(at)
        assert any("model exploded" in e.value for e in at.error)
        assert _flags(at, "guide_retry_btn")
        assert not _flags(at, "publish_protocol_btn")
        # Reruns do not silently restart a failed generation.
        at = at.run().run()
        assert guide_jobs.jobs.started == 1
    # The uploaded analysis is untouched and Retry produces the guide.
    assert at.session_state["pipeline_result"] is not None
    at = _click(at, "guide_retry_btn")
    at = _generated(at)
    assert guide_jobs.jobs.started == 2
    assert len(at.session_state["patient_guide"].all_items()) == 4
    assert not at.error


def test_job_manager_reuses_active_job_for_same_key():
    gate = threading.Event()
    calls = []

    def runner(advance):
        calls.append(1)
        gate.wait(5)
        advance()
        return "guide"

    first = guide_jobs.jobs.submit("k", runner, 1)
    second = guide_jobs.jobs.submit("k", runner, 1)
    assert first == second
    gate.set()
    assert guide_jobs.jobs.wait(first)
    assert len(calls) == 1
    job = guide_jobs.jobs.get(first)
    assert job.status == guide_jobs.COMPLETED and job.result == "guide"
    assert job.fraction == 1.0


# --- Task 2: decisions and publication ------------------------------------


def _mixed_guide():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    reject_item(_item(guide, "Follow-up"))
    edited = _item(guide, "inclusion condition 2")
    assert edit_item(edited, "Participants must meet inclusion condition 2 in full.") == []
    approve_item(edited)
    return result, guide


def test_review_states():
    _, guide = _mixed_guide()
    assert review_state(_item(guide, "Follow-up")) == "rejected"
    assert review_state(_item(guide, "inclusion condition 2")) == "edited"
    assert review_state(_item(guide, "between 18 and 65")) == "approved"
    _item(guide, "between 18 and 65").approved = False
    assert review_state(_item(guide, "between 18 and 65")) == "pending"
    assert state_counts(guide) == {
        "all": 8, "pending": 1, "approved": 5, "edited": 1, "rejected": 1
    }


def test_mix_of_approved_edited_and_rejected_publishes_without_the_rejected():
    result, guide = _mixed_guide()
    assert publish_block_reason(guide, result) is None
    at = _prepare_review_app(result, guide)
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")
    snapshot = at.session_state["published_protocols"]["protocol-eight"]["guide"]
    dump = snapshot.model_dump_json()
    assert "Follow-up" not in dump and "Day 30" not in dump
    assert "inclusion condition 2 in full" in dump
    assert len(snapshot.all_items()) == 7
    # The coordinator's own copy keeps the rejected statement for audit.
    assert len(at.session_state["patient_guide"].all_items()) == 8
    assert _item(at.session_state["patient_guide"], "Follow-up").rejected


def test_pending_statements_block_publication():
    result, guide = _mixed_guide()
    _item(guide, "between 18 and 65").approved = False
    reason = publish_block_reason(guide, result)
    assert "1 statement(s) still need a coordinator decision" in reason
    at = _prepare_review_app(result, guide)
    assert at.button(key="publish_protocol_btn").disabled
    assert at.session_state["published_protocols"] == {}


def test_every_statement_rejected_blocks_publication():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    for item in guide.all_items():
        reject_item(item)
    assert "Approve at least one" in publish_block_reason(guide, result)


@pytest.mark.parametrize(
    "marker,section",
    [("exclusion condition", "Who may not be able to join"), ("safety", "Safety information")],
)
def test_missing_mandatory_safety_section_blocks_publication(marker, section):
    result = _eight_statement_result()
    guide = _approved_guide(result)
    target = guide.who_cannot_join if "exclusion" in marker else guide.safety
    for item in target.items:
        reject_item(item)
    reason = publish_block_reason(guide, result)
    assert section in reason and "required safety disclosure" in reason
    # One approved statement in that section restores publishability.
    target.items[0].rejected = False
    target.items[0].approved = True
    target.items[0].needs_review = False
    assert publish_block_reason(guide, result) is None


def test_edited_statement_must_pass_verification_before_approval_or_publication():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    item = _item(guide, "inclusion condition 2")
    warnings = edit_item(item, "Participants must meet the second condition.")
    assert warnings
    at = _prepare_review_app(result, guide)
    assert at.button(key="guide_0_1_approve").disabled
    assert at.button(key="publish_protocol_btn").disabled
    # Even if approval were forced, publication still refuses it.
    item.approved = True
    assert "failed verification" in decision_block_reason(guide)


def test_statement_that_points_at_a_rejected_neighbour_blocks_publication():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    _item(guide, "inclusion condition 3").plain_text = (
        "Participants must also meet the conditions, as described above."
    )
    reject_item(_item(guide, "inclusion condition 2"))
    assert "refers to other content" in publish_block_reason(guide, result)


def test_changing_a_decision_updates_publishability():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    at = _prepare_review_app(result, guide)
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "guide_2_0_reject")  # visits statement
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "guide_2_0_reconsider")
    assert at.button(key="publish_protocol_btn").disabled
    assert any("need a coordinator decision" in w.value for w in at.warning)
    at = _click(at, "guide_2_0_approve")
    assert not at.button(key="publish_protocol_btn").disabled


def test_republishing_after_rejecting_more_removes_it_from_patients():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    at = _click(_prepare_review_app(result, guide), "publish_protocol_btn")
    assert "Day 30" in at.session_state["published_protocols"]["protocol-eight"][
        "guide"
    ].model_dump_json()
    at = _click(at, "guide_2_0_reject")
    assert any("earlier version" in i.value for i in at.info)
    at = _click(at, "publish_protocol_btn")
    assert "Day 30" not in at.session_state["published_protocols"]["protocol-eight"][
        "guide"
    ].model_dump_json()


# --- Rejected content never reaches patients ------------------------------


def _snapshot_without_followup():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    reject_item(_item(guide, "Follow-up"))
    return guide, build_published_snapshot(guide)


def test_snapshot_removes_rejected_statements_and_their_glossary_terms():
    guide = _approved_guide(_eight_statement_result())
    reject_item(_item(guide, "eGFR"))
    snapshot = build_published_snapshot(guide)
    assert "eGFR" not in snapshot.model_dump_json()
    assert all(t.term.casefold() != "egfr" for t in snapshot.glossary)
    assert not any(i.rejected for i in snapshot.all_items())
    assert [i for i in guide.all_items() if i.rejected]  # source guide untouched


def test_assistant_cannot_answer_from_a_rejected_statement():
    _, snapshot = _snapshot_without_followup()
    answer = with_coordinator_referral(
        answer_patient_from_guide("Is there a follow-up visit on Day 30?", snapshot)
    )
    assert "Day 30" not in answer.answer_text
    assert "Follow-up" not in answer.answer_text
    assert not answer.citations
    assert WITHHELD_MESSAGE in answer.answer_text


def test_assistant_ignores_rejected_items_even_in_an_unfiltered_guide():
    guide, _ = _snapshot_without_followup()
    visible = patient_visible_guide(guide)
    assert not any(i.rejected for i in visible.all_items())
    answer = answer_patient_from_guide("What visits are there?", visible)
    assert "Day 30" not in answer.answer_text and not answer.citations


def test_patient_portal_gives_safe_referral_for_rejected_topic():
    _, snapshot = _snapshot_without_followup()
    at = _patient_app({"p1": {"name": "Diabetes Study", "guide": snapshot}})
    at = _ask(at, "Is there a follow-up visit on Day 30?")
    answer = at.session_state["patient_history"][-1]["answer"]
    assert "Day 30" not in answer.answer_text
    assert WITHHELD_MESSAGE in answer.answer_text
    assert not answer.citations


def test_rejected_glossary_term_is_not_defined_to_patients():
    guide = _approved_guide(_eight_statement_result())
    reject_item(_item(guide, "eGFR"))
    at = _patient_app(
        {"p1": {"name": "Diabetes Study", "guide": build_published_snapshot(guide)}}
    )
    at = _ask(at, "What does eGFR mean?")
    assert at.session_state["patient_history"][-1]["answer"].intent != "term_definition"


def test_unfiltered_guide_with_rejection_is_not_a_published_protocol():
    guide = _approved_guide(_eight_statement_result())
    reject_item(_item(guide, "Follow-up"))
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["published_protocols"] = {"p1": {"name": "Leaky", "guide": guide}}
    at = _as(at, "patient").run()
    assert any("No published clinical trial protocols" in i.value for i in at.info)


def test_existing_patient_features_still_work_after_publishing_with_rejections():
    _, snapshot = _snapshot_without_followup()
    at = _patient_app({"p1": {"name": "Diabetes Study", "guide": snapshot}})
    at = _ask(at, "Who can join this study?")
    answer = at.session_state["patient_history"][-1]["answer"]
    assert answer.citations and answer.disclaimers
    at = _ask(at, "Can I stop taking my medication?")
    assert at.session_state["patient_history"][-1]["answer"].intent == "medical_advice_refused"
    at = _ask(at, "hi")
    assert at.session_state["patient_history"][-1]["answer"].intent == "conversational"
    at = _click(at, "new_chat_btn")
    assert at.session_state["patient_history"] == []


# --- Task 3: filters -------------------------------------------------------


def _cards(at):
    return [b.key for b in at.button if b.key and b.key.endswith("_save")]


def test_filters_replace_the_checkbox_and_show_counts():
    result, guide = _mixed_guide()
    _item(guide, "between 18 and 65").approved = False
    at = _prepare_review_app(result, guide)
    assert not at.toggle and not at.checkbox
    labels = {b.key: b.label for b in at.button if b.key and b.key.startswith("guide_filter_")}
    assert labels == {
        "guide_filter_all": "All Statements (8)",
        "guide_filter_pending": "Pending (1)",
        "guide_filter_approved": "Approved (5)",
        "guide_filter_edited": "Edited (1)",
        "guide_filter_rejected": "Rejected (1)",
    }
    assert at.button(key="guide_filter_all").proto.type == "primary"
    assert len(_cards(at)) == 8


@pytest.mark.parametrize(
    "key,expected", [("pending", 1), ("approved", 5), ("edited", 1), ("rejected", 1), ("all", 8)]
)
def test_each_filter_shows_only_its_statements(key, expected):
    result, guide = _mixed_guide()
    _item(guide, "between 18 and 65").approved = False
    at = _click(_prepare_review_app(result, guide), f"guide_filter_{key}")
    assert len(_cards(at)) == expected
    assert at.button(key=f"guide_filter_{key}").proto.type == "primary"
    others = [k for k in ("all", "pending", "approved", "edited", "rejected") if k != key]
    assert all(at.button(key=f"guide_filter_{o}").proto.type == "secondary" for o in others)


def test_counts_update_immediately_and_rejected_stay_visible_for_audit():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    at = _prepare_review_app(result, guide)
    assert at.button(key="guide_filter_rejected").label == "Rejected (0)"
    at = _click(at, "guide_2_0_reject")
    assert at.button(key="guide_filter_rejected").label == "Rejected (1)"
    assert at.button(key="guide_filter_approved").label == "Approved (7)"
    at = _click(at, "guide_filter_rejected")
    assert _cards(at) == ["guide_2_0_save"]
    assert any("Follow-up" in str(h.value) for h in at.get("html"))


def test_switching_filters_keeps_edits_and_decisions():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    guide.all_items()[1].approved = False
    at = _prepare_review_app(result, guide)
    at = _click(at, "guide_2_0_reject")
    for key in ("pending", "rejected", "approved", "all"):
        at = _click(at, f"guide_filter_{key}")
    saved = at.session_state["patient_guide"]
    assert saved.all_items()[1].approved is False
    assert _item(saved, "Follow-up").rejected
    assert state_counts(saved)["rejected"] == 1
    assert at.button(key="guide_filter_all").proto.type == "primary"
