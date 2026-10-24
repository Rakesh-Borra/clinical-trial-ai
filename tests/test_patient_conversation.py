"""Typo-tolerant, safe Patient Assistant conversation tests."""

import pytest

from tests.publish_helpers import generate_verified_guide
from agents.guide_patient_agent import answer_patient_from_guide
from agents.plain_language_agent import approve_item, generate_patient_guide
from models.protocol_models import (
    EligibilityCriterion,
    ProtocolAnalysis,
    TrialProcedure,
    TrialVisit,
)
from services.chat_intents import correct_query_spelling, social_reply_text


def _rewriter(original, section, feedback=None):
    return f"In simple words: {original}"


def _guide(approve=True):
    analysis = ProtocolAnalysis(
        trial_title="Study",
        inclusion_criteria=[
            EligibilityCriterion(description="Participants must be 18 or older.", source_page=1)
        ],
        visits=[TrialVisit(visit_name="Follow-up Visit", timing="Day 30", source_page=2)],
        procedures=[TrialProcedure(name="Blood pressure measurement", source_page=3)],
        analysis_complete=True,
    )
    guide = generate_verified_guide(analysis, rewriter=_rewriter)
    if approve:
        for item in guide.all_items():
            approve_item(item)
    return guide


@pytest.mark.parametrize("text", [
    "hello", "helllo", "hellllo", "hiii", "heyyy", "hi there", "helo",
])
def test_misspelled_greetings_get_greeting_without_retrieval(text):
    answer = answer_patient_from_guide(text, _guide())
    assert answer.intent == "conversational"
    assert not answer.citations
    assert "eligib" not in answer.answer_text.lower()
    assert "18" not in answer.answer_text


@pytest.mark.parametrize("text,word", [
    ("thx", "welcome"), ("thnks", "welcome"), ("thankyou", "welcome"),
    ("byee", "goodbye"), ("good mornin", "morning"), ("okay", "great"),
])
def test_thanks_goodbye_and_acknowledgements(text, word):
    answer = answer_patient_from_guide(text, _guide())
    assert answer.intent == "conversational"
    assert word in answer.answer_text.lower()
    assert not answer.citations


def test_social_matching_does_not_catch_real_words():
    for text in ("they", "this", "hell no", "what is a visit"):
        assert social_reply_text(text) is None


def test_social_works_even_when_nothing_is_published():
    answer = answer_patient_from_guide("helllo", _guide(approve=False))
    assert answer.intent == "conversational"


def test_spelling_correction_of_common_mistakes():
    assert correct_query_spelling("am i eligble") == "am i eligible"
    assert correct_query_spelling("what blud tests") == "what blood tests"
    assert correct_query_spelling("my next vist?") == "my next visit?"
    assert correct_query_spelling("what is a visit") == "what is a visit"


def test_misspelled_eligibility_decision_is_still_refused():
    answer = answer_patient_from_guide("am i eligble for this study", _guide())
    assert answer.intent == "eligibility_decision_refused"


def test_misspelled_visit_question_uses_approved_cited_content():
    answer = answer_patient_from_guide("when is my next vist", _guide())
    assert answer.citations
    assert {c.source_page for c in answer.citations} == {2}
    assert "Day 30" in answer.answer_text or "Follow-up" in answer.answer_text


def test_misspelled_blood_question_does_not_return_eligibility():
    answer = answer_patient_from_guide("do I have blud tests", _guide())
    assert "18 or older" not in answer.answer_text


def test_medical_advice_refusal_preserved_with_typos_and_greeting():
    guide = _guide()
    for text in ("should i stop my medcation", "helllo should I stop my medication"):
        answer = answer_patient_from_guide(text, guide)
        assert answer.intent == "medical_advice_refused"
        assert not answer.citations


def test_unpublished_content_is_never_used():
    guide = _guide(approve=False)
    for text in ("when is my next vist", "what are the visits", "who can join"):
        answer = answer_patient_from_guide(text, guide)
        assert not answer.citations
        assert "Day 30" not in answer.answer_text


def test_rejected_item_never_appears():
    guide = _guide()
    visit = guide.visits.items[0]
    visit.approved, visit.rejected = False, True
    answer = answer_patient_from_guide("when is my next vist", guide)
    assert "Day 30" not in answer.answer_text
    assert all("Day 30" not in c.statement for c in answer.citations)


def _history(guide, query):
    return [{"query": query, "answer": answer_patient_from_guide(query, guide)}]


def test_tell_me_more_adds_no_new_facts_and_points_to_study_team():
    guide = _guide()
    history = _history(guide, "what are the visits")
    answer = answer_patient_from_guide("tell me more", guide, history=history)
    assert not answer.citations
    assert "study team" in answer.answer_text.lower()


def test_ambiguous_input_without_context_asks_for_clarification():
    answer = answer_patient_from_guide("what about that", _guide(), history=[])
    assert answer.intent == "unsupported"
    assert "tell me a little more" in answer.answer_text.lower()
    assert not answer.citations


def test_greeting_after_a_topic_does_not_leak_previous_answer():
    guide = _guide()
    history = _history(guide, "who can join")
    answer = answer_patient_from_guide("hellllo", guide, history=history)
    assert answer.intent == "conversational"
    assert "18" not in answer.answer_text


def test_unrelated_gibberish_gets_no_clinical_facts():
    answer = answer_patient_from_guide("zxqv blorp", _guide())
    assert not answer.citations
    assert "18 or older" not in answer.answer_text
