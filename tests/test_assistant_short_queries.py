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


def _ask_agent(query, guide=None, history=None):
    return with_coordinator_referral(
        answer_patient_from_guide(query, guide or _guide(), history=history)
    )


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


@pytest.mark.parametrize(
    "query",
    [
        "Will they take blood from me?",
        "Will they collect blood from me?",
        "Do they draw a sample of my blood?",
        "Will I have to give a blood sample?",
        "Is a blood sample collected?",
    ],
)
def test_blood_test_and_collection_questions_use_approved_facts_only(query):
    answer = _ask_agent(query)
    assert GLUCOSE in _cited(answer)
    assert "does not specify whether a blood sample will be drawn" in answer.answer_text
    assert "coordinator" in answer.answer_text
    assert "from a vein" not in answer.answer_text


@pytest.mark.parametrize(
    "query",
    ["Do we have a blood test?", "What blood tests will I have?"],
)
def test_general_blood_test_questions_keep_their_existing_answer(query):
    answer = _ask_agent(query)

    assert GLUCOSE in _cited(answer)
    assert "does not specify whether a blood sample will be drawn" not in answer.answer_text


def test_explicit_approved_collection_statement_can_be_shared():
    guide = _guide()
    collection = "A blood sample will be collected at the screening visit."
    _add(guide, collection, page=4)

    answer = _ask_agent("Will they take blood from me?", guide)

    assert _cited(answer) == {collection}
    assert "screening visit" in answer.answer_text
    assert "does not specify whether" not in answer.answer_text


def test_blood_question_without_approved_blood_details_states_uncertainty():
    guide = _approved_guide(_eight_statement_result())

    answer = _ask_agent("Will they collect blood from me?", guide)

    assert answer.citations == []
    assert "does not specify whether a blood sample will be drawn" in answer.answer_text
    assert "coordinator" in answer.answer_text


def test_ambiguous_blood_test_question_does_not_claim_blood_is_drawn():
    answer = _ask_agent("Will they test my blood?")

    assert _cited(answer) == {GLUCOSE}
    assert "does not specify whether a blood sample will be drawn" in answer.answer_text
    assert "blood pressure" not in answer.answer_text


def test_pending_and_rejected_collection_statements_are_not_shared():
    guide = _guide()
    pending = _add(
        guide,
        "A blood sample will be collected during each visit.",
        approved=False,
    )

    answer = _ask_agent("Will they take blood from me?", guide)

    assert pending.plain_text not in answer.answer_text
    assert pending.plain_text not in _cited(answer)
    assert "does not specify whether a blood sample will be drawn" in answer.answer_text

    pending.approved = True
    pending.rejected = True
    answer = _ask_agent("Will they take blood from me?", guide)
    assert pending.plain_text not in answer.answer_text
    assert pending.plain_text not in _cited(answer)


def test_blood_follow_up_uses_last_cited_topic_and_preserves_safety_refusal():
    guide = _guide()
    prior_answer = _ask_agent("What blood tests will I have?", guide)
    history = [{"query": "What blood tests will I have?", "answer": prior_answer}]

    follow_up = _ask_agent("Will they take it?", guide, history=history)
    assert _cited(follow_up) == {GLUCOSE}
    assert "does not specify whether a blood sample will be drawn" in follow_up.answer_text

    refusal = _ask_agent("Can I skip it?", guide, history=history)
    assert refusal.intent == "medical_advice_refused"
    assert refusal.citations == []


def test_patient_chat_passes_prior_cited_turn_to_follow_up():
    guide = _guide()
    at = _patient_app({"p1": {"name": "Study", "guide": guide}})
    at = _ask(at, "What blood tests will I have?")
    at = _ask(at, "Will they take it?")

    answer = at.session_state["patient_history"][-1]["answer"]
    assert _cited(answer) == {GLUCOSE}
    assert "does not specify whether a blood sample will be drawn" in answer.answer_text


def test_unrelated_follow_up_does_not_inherit_blood_context():
    guide = _guide()
    history = [{"query": "What blood tests will I have?", "answer": _ask_agent("What blood tests will I have?", guide)}]

    answer = _ask_agent("What is the weather?", guide, history=history)

    assert answer.intent == "unsupported"
    assert not answer.citations


def test_timing_is_answered_only_when_the_guide_states_it():
    with_timing = _ask_agent("When is my blood test?")
    assert "each study visit" in with_timing.answer_text
    assert "doesn't say exactly when" not in with_timing.answer_text
    without = _ask_agent("When is my blood glucose test?", _guide(timing=False))
    assert "doesn't say exactly when" in without.answer_text
    assert "coordinator" in without.answer_text
    collection_uncertainty = _ask_agent(
        "When will my blood be taken?",
        _guide(timing=False),
    )
    assert "does not specify whether a blood sample will be drawn" in collection_uncertainty.answer_text
    assert "coordinator" in collection_uncertainty.answer_text


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
