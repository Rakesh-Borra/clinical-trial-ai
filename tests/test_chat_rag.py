"""Conversational RAG tests: social intents, typos, follow-ups, grounding, naturalize."""

from unittest.mock import patch

import pytest

from agents.guide_patient_agent import answer_patient_from_guide
from agents.plain_language_agent import approve_item
from services import chat_grounding
from services.chat_grounding import SAFE_UNGROUNDED_REPLY, naturalize, validate_grounding
from tests.test_guide_patient_agent import _approve_age_only, _guide


def _all_approved():
    guide = _guide()
    for section in (guide.who_can_join, guide.who_cannot_join, guide.visits, guide.safety):
        for item in section.items:
            approve_item(item)
    return guide


def _entry(answer):
    return {"query": "x", "answer": answer}


@pytest.mark.parametrize("msg", ["hello", "helllo", "hellllo", "hi", "thx", "thanks!", "bye", "ok cool"])
def test_social_messages_no_citations(msg):
    answer = answer_patient_from_guide(msg, _all_approved())
    assert answer.intent == "conversational"
    assert not answer.citations


@pytest.mark.parametrize("msg", ["am I eligble?", "how many vists are there", "what are the risks"])
def test_misspelled_questions_answered_or_refused_safely(msg):
    answer = answer_patient_from_guide(msg, _all_approved())
    assert answer.intent != "conversational"
    assert answer.answer_text


@pytest.mark.parametrize("msg", ["will it hurt", "can I quit whenever", "does the study cost anything", "who do I call"])
def test_out_of_scope_topics_have_no_citations(msg):
    answer = answer_patient_from_guide(msg, _all_approved())
    assert not answer.citations
    assert "study team" in answer.answer_text.lower()


def test_topic_change_does_not_leak_previous_topic():
    guide = _all_approved()
    first = answer_patient_from_guide("tell me about the visits", guide)
    assert first.citations
    second = answer_patient_from_guide("will it hurt", guide, [_entry(first)])
    assert not second.citations


def test_ordinal_follow_up_restates_nth_citation():
    guide = _all_approved()
    first = answer_patient_from_guide("what are the visits", guide)
    second = answer_patient_from_guide("what about the first one", guide, [_entry(first)])
    assert [c.statement for c in second.citations] == [first.citations[0].statement]


def test_ordinal_without_context_asks_clarification():
    answer = answer_patient_from_guide("what about the second one", _all_approved())
    assert not answer.citations
    assert "which one" in answer.answer_text.lower()


def test_unpublished_content_never_used():
    guide = _approve_age_only(_guide())
    answer = answer_patient_from_guide("what are the visits", guide)
    assert not answer.citations
    assert "Follow-up" not in answer.answer_text


def test_medical_advice_still_refused():
    answer = answer_patient_from_guide("should I stop taking my medication", _all_approved())
    assert answer.intent == "medical_advice_refused"
    assert not answer.citations


def test_validate_grounding_rejects_unapproved_citation():
    guide = _all_approved()
    answer = answer_patient_from_guide("what are the visits", guide)
    for item in guide.visits.items:
        item.approved = False
    checked = validate_grounding(answer, guide)
    assert checked.answer_text == SAFE_UNGROUNDED_REPLY
    assert not checked.citations


def test_validate_grounding_rejects_invented_numbers():
    guide = _all_approved()
    answer = answer_patient_from_guide("what are the visits", guide)
    bad = answer.model_copy(update={"answer_text": answer.answer_text + "\nYou will get 999 mg."})
    assert validate_grounding(bad, guide).answer_text == SAFE_UNGROUNDED_REPLY


def test_validate_grounding_passes_valid_answer():
    guide = _all_approved()
    answer = answer_patient_from_guide("what are the visits", guide)
    assert validate_grounding(answer, guide) == answer


def test_naturalize_off_by_default_makes_no_model_call(monkeypatch):
    monkeypatch.delenv("CTAI_CHAT_LLM", raising=False)
    answer = answer_patient_from_guide("what are the visits", _all_approved())
    with patch.object(chat_grounding, "chat_completion") as mock:
        assert naturalize("what are the visits", answer) == answer
        mock.assert_not_called()


def _llm_reply(text):
    class R:
        reply = text
    return R()


def test_naturalize_accepts_valid_and_rejects_new_numbers(monkeypatch):
    monkeypatch.setenv("CTAI_CHAT_LLM", "1")
    answer = answer_patient_from_guide("what are the visits", _all_approved())
    with patch.object(chat_grounding, "chat_completion", return_value=_llm_reply("There are visits at Day 30.")):
        out = naturalize("what are the visits", answer)
        assert "Day 30" in out.answer_text
    with patch.object(chat_grounding, "chat_completion", return_value=_llm_reply("You have 77 visits.")):
        assert naturalize("what are the visits", answer) == answer
    with patch.object(chat_grounding, "chat_completion", return_value=_llm_reply("You should take more medication.")):
        assert naturalize("what are the visits", answer) == answer
    with patch.object(chat_grounding, "chat_completion", side_effect=RuntimeError("down")):
        assert naturalize("what are the visits", answer) == answer


def test_naturalize_keeps_deterministic_answer_on_insufficient_prefix(monkeypatch):
    monkeypatch.setenv("CTAI_CHAT_LLM", "1")
    answer = answer_patient_from_guide("what are the visits", _all_approved())
    with patch.object(
        chat_grounding,
        "chat_completion",
        return_value=_llm_reply("Not enough information in the approved study guide."),
    ):
        assert naturalize("what are the visits", answer) == answer


def test_naturalize_passes_follow_up_history_into_prompt(monkeypatch):
    monkeypatch.setenv("CTAI_CHAT_LLM", "1")
    guide = _all_approved()
    first = answer_patient_from_guide("what are the visits", guide)
    captured = {}

    def _fake_completion(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return _llm_reply("The first visit is the follow-up on Day 30.")

    with patch.object(chat_grounding, "chat_completion", side_effect=_fake_completion):
        naturalize(
            "what about the first one",
            first,
            history=[{"query": "what are the visits", "answer": first}],
        )
    assert "Recent conversation" in captured["prompt"]
    assert "what are the visits" in captured["prompt"]


def test_naturalize_never_rewrites_refusals(monkeypatch):
    monkeypatch.setenv("CTAI_CHAT_LLM", "1")
    answer = answer_patient_from_guide("should I stop my medication", _all_approved())
    with patch.object(chat_grounding, "chat_completion") as mock:
        assert naturalize("should I stop my medication", answer) == answer
        mock.assert_not_called()


def test_repeated_identical_answer_is_not_prefixed_unless_asked():
    guide = _all_approved()
    first = answer_patient_from_guide("what are the risks", guide)
    again = answer_patient_from_guide("what are the risks", guide, [_entry(first)])
    assert not again.answer_text.startswith("Here is that again")
    assert again.citations == first.citations

    repeated = answer_patient_from_guide(
        "say that again",
        guide,
        [{"query": "what are the risks", "answer": first}],
    )
    assert repeated.answer_text.startswith("Here is that again")
    assert repeated.citations == first.citations
