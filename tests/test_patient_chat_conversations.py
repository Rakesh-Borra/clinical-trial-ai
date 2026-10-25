"""Regression tests for conversational patient chat (Task 2 sample dialogues)."""

from __future__ import annotations

from models.plain_language_models import GuideItem, GuideSection, PatientGuide
from agents.guide_patient_agent import answer_patient_from_guide
from services.chat_grounding import SAFE_UNGROUNDED_REPLY, naturalize
from services.guide_review import WITHHELD_MESSAGE
from unittest.mock import patch
import services.chat_grounding as chat_grounding


def _item(section: str, plain: str, page: int, item_id: str) -> GuideItem:
    return GuideItem(
        section=section,  # type: ignore[arg-type]
        original_text=plain,
        plain_text=plain,
        source_page=page,
        approved=True,
        rejected=False,
        item_id=item_id,
        source_status="verified",
        analysis_complete=True,
    )


def _section(title: str, items: list[GuideItem]) -> GuideSection:
    return GuideSection(title=title, introduction="", items=items)


def t1d_like_guide() -> PatientGuide:
    """Mirrors the eight approved statements published for T1D protocol 2."""
    return PatientGuide(
        trial_title="T1D protocol 2",
        who_can_join=_section(
            "Who may be able to join",
            [
                _item("inclusion_criteria", "People in this trial must be between 18 and 65 years old.", 1, "age"),
                _item("inclusion_criteria", "People in this trial must have Type 2 Diabetes.", 1, "t2d"),
            ],
        ),
        who_cannot_join=_section(
            "Who may not be able to join",
            [
                _item(
                    "exclusion_criteria",
                    "Participants with severe kidney disease are excluded.",
                    1,
                    "kidney",
                )
            ],
        ),
        visits=_section(
            "Trial visits",
            [
                _item("visits", "Screening Visit. Timing: Day 0", 2, "screen"),
                _item("visits", "This trial has a follow-up visit on Day 30.", 2, "fu"),
            ],
        ),
        procedures=_section(
            "Tests and procedures",
            [
                _item("procedures", "Blood glucose testing", 2, "glucose"),
                _item("procedures", "Blood pressure measurement", 2, "bp"),
            ],
        ),
        safety=_section(
            "Safety information",
            [
                _item(
                    "safety_information",
                    "We will watch for any side effects that happen during the trial.",
                    2,
                    "safety",
                )
            ],
        ),
        disclaimer="General study information only.",
        analysis_complete=True,
    )


def _turn(query: str, answer):
    return {"query": query, "answer": answer}


def test_arrival_statements_get_clarifying_welcome():
    guide = t1d_like_guide()
    for msg in ("I came for test", "I'm here for the study", "I am here for a visit"):
        answer = answer_patient_from_guide(msg, guide)
        assert answer.intent == "conversational"
        assert not answer.citations
        assert "welcome" in answer.answer_text.lower()
        assert "doesn't specify" not in answer.answer_text.lower()
        assert "approved" in answer.answer_text.lower()


def test_what_happens_during_study_summarizes_visits_procedures_and_safety():
    guide = t1d_like_guide()
    answer = answer_patient_from_guide("What happens during the study?", guide)
    text = answer.answer_text
    lower = text.lower()
    assert answer.citations
    assert "- **Study visits:**" in text
    assert "- **Tests:**" in text
    assert "- **Safety:**" in text
    assert "Source: Protocol page" in text
    assert "day 0" in lower or "screening" in lower
    assert "day 30" in lower or "follow-up" in lower
    assert "glucose" in lower
    assert "blood pressure" in lower or "pressure" in lower
    assert "side effect" in lower
    assert len(answer.citations) >= 3


def test_conversation_a_timeline_follow_ups_are_careful():
    guide = t1d_like_guide()
    first = answer_patient_from_guide("What happens during the study?", guide)
    second = answer_patient_from_guide(
        "How long does the study last?",
        guide,
        [_turn("What happens during the study?", first)],
    )
    third = answer_patient_from_guide(
        "When does this end?",
        guide,
        [
            _turn("What happens during the study?", first),
            _turn("How long does the study last?", second),
        ],
    )
    for answer in (second, third):
        text = answer.answer_text.lower()
        assert answer.citations
        assert "day 30" in text or "follow-up" in text
        assert "officially ends" in text or "does not specify" in text
        assert "coordinator" in text
        assert "study ends on day 30" not in text
        assert "here is that again" not in text
    # Follow-up should stay brief rather than restating the full overview.
    assert "- **Study visits:**" not in third.answer_text


def test_narrow_tests_question_stays_concise_with_citation():
    answer = answer_patient_from_guide("What tests will I have?", t1d_like_guide())
    assert "Blood glucose testing" in answer.answer_text
    assert "Blood pressure measurement" in answer.answer_text
    assert "Source: Protocol page" in answer.answer_text
    assert answer.citations


def test_am_i_eligible_refuses_personal_decision():
    answer = answer_patient_from_guide("Am I eligible?", t1d_like_guide())
    assert answer.intent == "eligibility_decision_refused"
    assert "personally" in answer.answer_text.lower() or "study team" in answer.answer_text.lower()


def test_when_does_this_end_without_history_uses_visit_timeline():
    answer = answer_patient_from_guide("When does this end?", t1d_like_guide())
    assert answer.citations
    assert "day 30" in answer.answer_text.lower()
    assert "officially ends" in answer.answer_text.lower() or "does not say" in answer.answer_text.lower()
    assert "i'm not sure i understood" not in answer.answer_text.lower()


def test_when_does_this_test_end_asks_or_explains_missing_duration():
    guide = t1d_like_guide()
    bare = answer_patient_from_guide("When does this test end?", guide)
    assert bare.intent == "clarification" or "not specified" in bare.answer_text.lower() or "coordinator" in bare.answer_text.lower()
    assert "i'm not sure i understood" not in bare.answer_text.lower()

    prior = answer_patient_from_guide("What tests will I have?", guide)
    follow = answer_patient_from_guide(
        "When does this test end?", guide, [_turn("What tests will I have?", prior)]
    )
    assert "glucose" in follow.answer_text.lower() or follow.citations
    assert "not specified" in follow.answer_text.lower() or "coordinator" in follow.answer_text.lower()


def test_affirmative_follow_up_after_tests_offer_explains_procedures():
    guide = t1d_like_guide()
    offer = answer_patient_from_guide("I came for test", guide)
    assert "tests involved" in offer.answer_text.lower()
    for reply in ("Explain", "Yes", "Tell me", "Go ahead"):
        answer = answer_patient_from_guide(reply, guide, [_turn("I came for test", offer)])
        lower = answer.answer_text.lower()
        assert answer.citations, reply
        assert answer.intent == "procedures", reply
        assert "glucose" in lower, reply
        assert "blood pressure" in lower or "pressure" in lower, reply
        assert "are you asking about the tests" not in lower, reply
        assert "source: protocol page" in lower, reply


def test_informal_testing_period_questions_use_visits_without_inventing_end():
    guide = t1d_like_guide()
    for msg in (
        "Any days?",
        "How many days does the test prolong?",
        "How long will testing continue?",
    ):
        answer = answer_patient_from_guide(msg, guide)
        lower = answer.answer_text.lower()
        assert answer.citations, msg
        assert "day 0" in lower or "screening" in lower, msg
        assert "day 30" in lower or "follow-up" in lower, msg
        assert "not specified in the provided protocol" in lower, msg
        assert "study ends on day 30" not in lower, msg
        assert "takes 30 minutes" not in lower, msg


def test_what_happens_after_that_uses_recent_context():
    guide = t1d_like_guide()
    offer = answer_patient_from_guide("I came for test", guide)
    explained = answer_patient_from_guide(
        "Explain", guide, [_turn("I came for test", offer)]
    )
    after = answer_patient_from_guide(
        "What happens after that?",
        guide,
        [
            _turn("I came for test", offer),
            _turn("Explain", explained),
        ],
    )
    lower = after.answer_text.lower()
    assert after.citations
    assert "day 0" in lower or "day 30" in lower or "follow-up" in lower
    assert "not specified" in lower or "coordinator" in lower


def test_unavailable_display_wording():
    from services.chat_intents import display_unavailable_message

    old = (
        "The trial document did not include this information. "
        "Please talk with your study team if you have questions."
    )
    shown = display_unavailable_message(old)
    assert shown == "Not specified in the provided protocol."
    assert "did not include this information" not in shown.lower()


def test_kind_of_testings_returns_approved_procedures():
    answer = answer_patient_from_guide(
        "What kind of testings do we have?", t1d_like_guide()
    )
    cited = {c.statement for c in answer.citations}
    assert "Blood glucose testing" in cited
    assert "Blood pressure measurement" in cited
    assert "don't have approved information that answers that question" not in answer.answer_text.lower()


def test_risks_and_join_guardrails_still_hold():
    guide = t1d_like_guide()
    risks = answer_patient_from_guide("What are the risks?", guide)
    assert risks.citations and "side effect" in risks.answer_text.lower()
    join = answer_patient_from_guide("Can I join?", guide)
    assert join.intent == "eligibility_decision_refused"
    assert "personally" in join.answer_text.lower()


def test_medical_advice_still_refused():
    answer = answer_patient_from_guide(
        "Should I stop my medication?", t1d_like_guide()
    )
    assert answer.intent == "medical_advice_refused"
    assert not answer.citations


def test_protocol_isolation_for_overview():
    other = PatientGuide(
        trial_title="Other study",
        who_can_join=_section("Who may be able to join", []),
        who_cannot_join=_section("Who may not be able to join", []),
        visits=_section(
            "Trial visits",
            [_item("visits", "Only Other study Friday MRI visit.", 9, "other")],
        ),
        procedures=_section("Tests and procedures", []),
        safety=_section("Safety information", []),
        disclaimer="General study information only.",
        analysis_complete=True,
    )
    answer = answer_patient_from_guide("What happens during the study?", other)
    assert all("Friday MRI" in c.statement for c in answer.citations)
    assert "Day 30" not in answer.answer_text


def test_naturalize_with_history_mocked(monkeypatch):
    monkeypatch.setenv("CTAI_CHAT_LLM", "1")
    guide = t1d_like_guide()
    first = answer_patient_from_guide("What happens during the study?", guide)
    captured = {}

    class R:
        reply = "You have a screening visit on Day 0 and a follow-up on Day 30."

    def fake(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return R()

    with patch.object(chat_grounding, "chat_completion", side_effect=fake):
        out = naturalize(
            "How long will that take?",
            first,
            history=[_turn("What happens during the study?", first)],
        )
    assert "Day 0" in out.answer_text
    assert "Recent conversation" in captured["prompt"]
    assert SAFE_UNGROUNDED_REPLY not in out.answer_text or out.citations
