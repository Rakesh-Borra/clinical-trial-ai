"""Short, natural patient questions and suggestion visibility."""

import pytest

from agents.guide_patient_agent import answer_patient_from_guide
from agents.plain_language_agent import reject_item
from models.plain_language_models import GuideItem
from services.guide_review import (
    WITHHELD_MESSAGE,
    build_published_snapshot,
    with_coordinator_referral,
)
from tests.test_patient_guide_ui import _approved_guide, _click, _eight_statement_result
from tests.test_patient_portal_layout import _patient_app

GLUCOSE = "Blood glucose will be tested at each study visit."
PRESSURE = "Your blood pressure will be checked at the screening visit."


def _add(guide, text, page=3, approved=True):
    item = GuideItem(
        section="procedures",
        original_text=text,
        plain_text=text,
        source_page=page,
        approved=approved,
    )
    guide.procedures.items.append(item)
    return item


def _guide(timing=True):
    guide = _approved_guide(_eight_statement_result())
    _add(guide, GLUCOSE if timing else "Blood glucose will be tested.")
    _add(guide, PRESSURE)
    return guide


def _ask_agent(query, guide=None):
    return with_coordinator_referral(answer_patient_from_guide(query, guide or _guide()))


def _cited(answer):
    return {c.statement for c in answer.citations}


def _ask(at, text):
    at.chat_input(key="patient_chat_input").set_value(text)
    return at.run()


def _suggestions(at):
    return [b for b in at.button if b.key and b.key.startswith("suggest_")]


@pytest.mark.parametrize("query", ["blood test", "blood tests", "bloodwork", "blood glucose", "blood glucose?", "gluclose test", "is my sugar tested"])
def test_glucose_queries_retrieve_only_the_approved_glucose_statement(query):
    answer = _ask_agent(query)
    assert _cited(answer) == {GLUCOSE}
    assert PRESSURE not in answer.answer_text
    assert "protocol page 3" in answer.answer_text


def test_blood_pressure_is_answered_separately():
    assert _cited(_ask_agent("blood pressure")) == {PRESSURE}


def test_bare_blood_asks_a_useful_clarification():
    answer = _ask_agent("blood")
    assert answer.intent == "clarification" and not answer.citations
    assert "blood glucose" in answer.answer_text and "blood pressure" in answer.answer_text
    assert WITHHELD_MESSAGE not in answer.answer_text


def test_general_test_question_lists_the_approved_tests():
    answer = _ask_agent("What tests will I have?")
    assert _cited(answer) == {GLUCOSE, PRESSURE}
    assert _cited(_ask_agent("What blood tests will I have?")) >= {GLUCOSE}


def test_timing_is_answered_only_when_the_guide_states_it():
    with_timing = _ask_agent("When is my blood test?")
    assert "each study visit" in with_timing.answer_text
    assert "doesn't say exactly when" not in with_timing.answer_text
    without = _ask_agent("When is my blood glucose test?", _guide(timing=False))
    assert "doesn't say exactly when" in without.answer_text
    assert "coordinator" in without.answer_text
    assert _ask_agent("When will my blood be taken?", _guide(timing=False)).intent == "clarification"


def test_unavailable_information_is_stated_not_invented():
    guide = _approved_guide(_eight_statement_result())  # no procedures at all
    for query in ("blood test", "blood glucose"):
        answer = _ask_agent(query, guide)
        assert not answer.citations
        assert "glucose" not in answer.answer_text.lower() or "doesn't include" in answer.answer_text


@pytest.mark.parametrize("query", ["Can I skip my blood test?", "Can I skip the blood test?", "can I refuse the blood test", "do I have to do the blood test", "hi, can I skip the blood test"])
def test_skipping_or_changing_procedures_stays_refused(query):
    answer = _ask_agent(query)
    assert answer.intent == "medical_advice_refused" and not answer.citations


@pytest.mark.parametrize("query", ["What is the weather", "parking cost", "who won the game", "blood moon festival"])
def test_unrelated_questions_do_not_match_blood_content(query):
    answer = _ask_agent(query)
    assert not answer.citations
    assert GLUCOSE not in answer.answer_text and PRESSURE not in answer.answer_text


def test_pending_and_rejected_statements_are_never_retrieved():
    guide = _approved_guide(_eight_statement_result())
    _add(guide, GLUCOSE, approved=False)
    rejected = _add(guide, PRESSURE)
    reject_item(rejected)
    for query in ("blood test", "blood glucose", "blood pressure", "blood"):
        answer = _ask_agent(query, guide)
        assert not answer.citations
        assert GLUCOSE not in answer.answer_text and PRESSURE not in answer.answer_text


def test_rejected_topic_gets_safe_referral_through_the_published_snapshot():
    guide = _guide()
    reject_item(next(i for i in guide.all_items() if i.plain_text == PRESSURE))
    snapshot = build_published_snapshot(guide)
    answer = _ask_agent("blood pressure", snapshot)
    assert not answer.citations and PRESSURE not in answer.answer_text
    assert WITHHELD_MESSAGE in answer.answer_text
    # The unrejected topic is unaffected.
    assert _cited(_ask_agent("blood glucose", snapshot)) == {GLUCOSE}


def test_greeting_and_mixed_questions_keep_their_paths():
    at = _patient_app({"p1": {"name": "Study", "guide": _guide()}})
    at = _ask(at, "hi")
    assert at.session_state["patient_history"][-1]["answer"].intent == "conversational"
    at = _ask(at, "hi, can I skip my blood test?")
    assert at.session_state["patient_history"][-1]["answer"].intent == "medical_advice_refused"


# --- suggested questions ----------------------------------------------------


def test_suggestions_show_on_empty_chat_and_hide_after_any_first_message():
    at = _patient_app()
    assert len(_suggestions(at)) == 5
    assert any("Try one of these" in c.value for c in at.caption)
    at = _ask(at, "hi")
    assert not _suggestions(at)
    assert not any("Try one of these" in c.value for c in at.caption)
    at = at.run()
    assert not _suggestions(at)


def test_suggestion_button_still_answers_then_hides_the_section():
    at = _patient_app()
    at.button(key="suggest_0").click()
    at = at.run()
    answer = at.session_state["patient_history"][-1]["answer"]
    assert answer.citations
    assert not _suggestions(at)


def test_new_chat_brings_suggestions_back_and_history_restores_without_them():
    at = _patient_app()
    at = _ask(at, "Who can join this study?")
    at = _click(at, "new_chat_btn")
    assert at.session_state["patient_history"] == []
    assert len(_suggestions(at)) == 5
    # Reopen the earlier conversation: it has messages, so no suggestions.
    chat_id = next(k for k in (b.key for b in at.button) if k.startswith("history_"))
    at = _click(at, chat_id)
    assert at.session_state["patient_history"]
    assert not _suggestions(at)
    assert at.session_state["patient_history"][0]["answer"].citations
    # And the fresh empty conversation does not lose the earlier one.
    at = _click(at, "new_chat_btn")
    assert len(_suggestions(at)) == 5
