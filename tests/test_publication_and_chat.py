import pytest
from streamlit.testing.v1 import AppTest

from services.chat_intents import social_reply_text
from tests.test_patient_guide_ui import (
    _approved_guide,
    _click,
    _eight_statement_result,
    _prepare_review_app,
)
from tests.test_patient_portal_layout import _patient_app
from ui import chat_store, patient_view, split_pane, theme

PUBLISHED = "Published successfully! Coordinator task completed."


def _published_review():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    at = _prepare_review_app(result, guide)
    return _click(at, "publish_protocol_btn"), guide


def _flatten(node):
    for child in node.children.values():
        yield child
        if hasattr(child, "children"):
            yield from _flatten(child)


def _ask(at, text):
    at.chat_input(key="patient_chat_input").set_value(text)
    return at.run()


def _switch(at, protocol_id):
    at.selectbox(key="patient_protocol_select").set_value(protocol_id)
    at = at.run()
    at.button(key="view_protocol_summary_btn").click()
    return at.run()


def _intents(at):
    return [m["answer"].intent for m in at.session_state["patient_history"]]


# --- Tasks 1 and 2: publication state and Preview -------------------------


def test_preview_hidden_before_publication():
    at = _prepare_review_app(_eight_statement_result(), _approved_guide())
    assert not at.success or all(PUBLISHED not in s.value for s in at.success)
    assert not [b for b in at.button if b.key == "preview_patient_btn"]


def test_preview_shown_after_publication_and_opens_that_protocol():
    at, _ = _published_review()
    assert [b for b in at.button if b.key == "preview_patient_btn"]
    at = _click(at, "preview_patient_btn")
    assert at.session_state["role"] == "patient"
    assert at.session_state["patient_protocol_select"] == "protocol-eight"


def test_preview_is_last_element_after_all_status_messages():
    at, _ = _published_review()
    elements = list(_flatten(at.main))
    keys = [getattr(e, "key", None) for e in elements]
    preview = keys.index("preview_patient_btn")
    publish = keys.index("publish_protocol_btn")
    success = next(
        i for i, e in enumerate(elements) if PUBLISHED in str(getattr(e, "value", ""))
    )
    assert publish < success < preview
    assert [k for k in keys[preview + 1 :] if k and k.endswith("_btn")] == []


def test_publication_shows_one_consistent_state():
    at, _ = _published_review()
    assert [s.value for s in at.success if PUBLISHED in s.value] == [PUBLISHED]
    assert not at.error
    assert not any("final approval validation" in w.value for w in at.warning)


def test_publication_state_survives_rerun_without_contradiction():
    at, _ = _published_review()
    at = at.run()
    assert [s.value for s in at.success if PUBLISHED in s.value] == [PUBLISHED]
    assert not at.error and not at.warning


def test_edits_after_publishing_do_not_claim_current_success():
    at, guide = _published_review()
    guide.all_items()[0].approved = False
    at = at.run()
    assert not any(PUBLISHED in s.value for s in at.success)
    assert any("earlier version" in i.value for i in at.info)
    assert any("still need a coordinator decision" in w.value for w in at.warning)
    assert [b for b in at.button if b.key == "preview_patient_btn"]
    assert at.button(key="publish_protocol_btn").disabled


def test_invalid_published_snapshot_is_not_reported_as_published():
    at, _ = _published_review()
    at.session_state["published_protocols"]["protocol-eight"]["guide"] = None
    at = at.run()
    assert not any(PUBLISHED in s.value for s in at.success)
    assert not [b for b in at.button if b.key == "preview_patient_btn"]


def test_validation_failure_still_blocks_publication():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    guide.all_items()[0].approved = False
    at = _prepare_review_app(result, guide)
    assert at.button(key="publish_protocol_btn").disabled
    assert at.session_state["published_protocols"] == {}
    assert not [b for b in at.button if b.key == "preview_patient_btn"]


# --- Task 3: New chat and chat history ------------------------------------


def test_new_chat_keeps_old_conversation_and_starts_empty():
    at = _patient_app()
    at.button(key="suggest_0").click()
    at = at.run()
    first = at.session_state["patient_history"][0]
    at = _click(at, "new_chat_btn")
    assert at.session_state["patient_history"] == []
    assert at.session_state["patient_protocol_select"] == "p1"
    assert [m.value for m in at.get("html") if "Hi! 👋" in str(m.value)]
    at.button(key="suggest_1").click()
    at = at.run()
    chats = at.session_state["patient_chats"]["p1"]
    assert len(chats) == 2
    assert chats[0]["messages"][0] is first


def test_new_chat_on_empty_conversation_creates_nothing():
    at = _click(_patient_app(), "new_chat_btn")
    assert not at.session_state["patient_history"]
    assert not any(at.session_state["patient_chats"].values()) if "patient_chats" in at.session_state else True


def test_history_lists_newest_first_and_restores_conversation_with_citations():
    at = _patient_app()
    at.button(key="suggest_0").click()
    at = at.run()
    original = at.session_state["patient_history"][0]
    at = _click(at, "new_chat_btn")
    at.button(key="suggest_2").click()
    at = at.run()
    ids = [c["id"] for c in at.session_state["patient_chats"]["p1"]]
    buttons = [b.key for b in at.button if b.key and b.key.startswith("history_")]
    assert buttons == [f"history_{i}" for i in reversed(ids)]
    at = _click(at, f"history_{ids[0]}")
    assert at.session_state["patient_history"][0] is original
    assert original["answer"].citations
    assert "protocol page" in original["answer"].answer_text.lower()
    assert any("Who can join" in m.value or original["query"] in m.value for m in at.markdown)


def test_chat_history_is_isolated_by_protocol():
    guide = _approved_guide()
    at = _patient_app(
        {
            "p1": {"name": "Study A", "guide": guide},
            "p2": {"name": "Study B", "guide": _approved_guide()},
        }
    )
    at.button(key="suggest_0").click()
    at = at.run()
    a_ids = [c["id"] for c in at.session_state["patient_chats"]["p1"]]
    at = _switch(at, "p2")
    assert at.session_state["patient_history"] == []
    assert not [b for b in at.button if b.key and b.key.startswith("history_")]
    at.button(key="suggest_1").click()
    at = at.run()
    assert len(at.session_state["patient_chats"]["p2"]) == 1
    assert len(at.session_state["patient_chats"]["p1"]) == 1
    at = _switch(at, "p1")
    keys = [b.key for b in at.button if b.key and b.key.startswith("history_")]
    assert keys == [f"history_{i}" for i in a_ids]


def test_history_ignores_conversations_from_another_guide_version():
    at = _patient_app()
    at.button(key="suggest_0").click()
    at = at.run()
    changed = _approved_guide()
    changed.trial_title = "Different approved content"
    assert chat_store.list_chats("p1", changed) == []


def test_chat_survives_rerun_and_panel_resize_controls_are_client_side():
    at = _patient_app()
    at.button(key="suggest_0").click()
    at = at.run().run()
    assert len(at.session_state["patient_history"]) == 1
    # Resize/minimize/maximize run in the browser only: no Python callbacks.
    for banned in ("setComponentValue", "postMessage", "fetch(", "XMLHttpRequest", "WebSocket"):
        assert banned not in split_pane._SCRIPT
    assert "<button" in split_pane.CONTROLS_HTML
    assert "ctai-ctl-min" in split_pane.CONTROLS_HTML


def test_toolbar_buttons_exist_beside_window_controls():
    at = _patient_app()
    assert at.button(key="new_chat_btn")
    css = theme.DARK_WORKSPACE_CSS + theme.CSS
    assert ".st-key-chat_toolbar" in css


# --- Task 4: greetings ------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hi", "Hi! 👋 Welcome to your study assistant. How can I help you today?"),
        ("Hello!", "Hi! 👋 Welcome to your study assistant. How can I help you today?"),
        (
            "good morning",
            "Good morning! ☀️ I'm here to help you understand your clinical trial. What would you like to know?",
        ),
        (
            "thank you",
            "You're welcome! Let me know if you have any other questions about your study.",
        ),
        ("Thanks!", "You're welcome! Let me know if you have any other questions about your study."),
    ],
)
def test_social_replies_match_spec(text, expected):
    assert social_reply_text(text) == expected


@pytest.mark.parametrize(
    "text",
    ["hey", "good afternoon", "Good evening", "how are you?", "hi, how are you?", "bye", "goodbye!"],
)
def test_other_social_messages_get_short_replies(text):
    reply = social_reply_text(text)
    assert reply and len(reply) < 140
    assert "medical advice" not in reply.lower()


@pytest.mark.parametrize(
    "text",
    [
        "Hi, can I stop taking my medication?",
        "Thanks, but am I eligible?",
        "hello what are the visits",
        "good morning doctor I have chest pain",
        "how are you feeling after the drug",
        "bye the way, who can join?",
        "thank you for the drug dose",
        "",
        "   ",
        "What visits are there?",
    ],
)
def test_mixed_or_other_messages_are_not_social(text):
    assert social_reply_text(text) is None


def test_greeting_in_chat_is_friendly_keeps_query_and_skips_the_agent(monkeypatch):
    at = _patient_app()

    def boom(*args, **kwargs):
        raise AssertionError("agent must not run for a greeting")

    monkeypatch.setattr(patient_view, "answer_patient_from_guide", boom)
    at = _ask(at, "hi")
    entry = at.session_state["patient_history"][0]
    assert entry["query"] == "hi"
    assert entry["answer"].answer_text.startswith("Hi! 👋")
    assert "not sure I understood" not in entry["answer"].answer_text
    assert entry["answer"].intent == "conversational"
    assert not entry["answer"].citations


def test_mixed_greeting_and_medical_question_uses_safety_handling():
    at = _patient_app()
    at = _ask(at, "Hi, can I stop taking my medication?")
    assert _intents(at) == ["medical_advice_refused"]
    at = _ask(at, "Thanks, but am I eligible?")
    assert _intents(at)[-1] == "eligibility_decision_refused"


def test_protocol_answers_and_fallback_unchanged_after_greeting():
    at = _patient_app()
    at = _ask(at, "hello")
    assert not [b for b in at.button if b.key.startswith("suggest_")]
    at = _ask(at, "Who can join this study?")
    answer = at.session_state["patient_history"][-1]["answer"]
    assert answer.citations and "protocol page" in answer.answer_text.lower()
    at = _ask(at, "blorp zinger")
    assert "not sure I understood" in at.session_state["patient_history"][-1]["answer"].answer_text
