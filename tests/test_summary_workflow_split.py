from tests.auth_helpers import sign_in_as
"""Summary generation lives in Step 2; Step 3 reviews. Readable formatting."""

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from services import review_store
from services.summary_review import (
    build_summary_prompt, extractive_summary, format_summary_markdown,
    is_summary_approved,
)
from tests.test_patient_guide_ui import (
    APP_PATH, _approved_guide, _click, _pipeline_result,
    _prepare_review_app,
)

LONG = " ".join(f"Sentence number {i} is here." for i in range(1, 11))


def _step2(guide=None):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator")
    at.session_state["coord_step"] = "2 · Summary"
    at.session_state["coord_max_step"] = 2
    at.session_state["pipeline_result"] = _pipeline_result()
    at.session_state["coordinator_protocol_id"] = "protocol-eight"
    at.session_state["coordinator_protocols"] = {"protocol-eight": {"name": "Eight"}}
    if guide is not None:
        at.session_state["patient_guide"] = guide
    return at.run()


def _keys(at):
    return {b.key for b in at.button}


def test_step3_without_summary_never_generates_and_directs_to_step2(tmp_path, monkeypatch):
    monkeypatch.setenv("CTAI_DB_PATH", str(tmp_path / "fresh.db"))
    with patch("services.summary_review.generate_patient_summary") as gen:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        sign_in_as(at.session_state, "coordinator")
        at.session_state["coord_step"] = "3 · Review"
        at.session_state["coord_max_step"] = 2
        at.session_state["pipeline_result"] = _pipeline_result()
        at.session_state["coordinator_protocol_id"] = "p-none"
        at.session_state["coordinator_protocols"] = {"p-none": {"name": "N"}}
        at.session_state["patient_guide"] = _approved_guide()
        at.run()
    assert "summary_generate_btn" not in _keys(at)
    assert not gen.called
    assert review_store.load_summary("p-none") is None
    assert any("Step 2" in i.value for i in at.info)


def test_step3_shows_existing_summary_and_keeps_edit_controls():
    at = _prepare_review_app(guide=_approved_guide())
    assert "summary_generate_btn" not in _keys(at)
    assert {"summary_save_btn", "summary_revert_btn"} <= _keys(at)
    assert review_store.load_summary("protocol-eight") is not None


def test_step2_retains_generation_and_requires_explicit_regeneration():
    at = _step2(guide=_approved_guide())
    assert review_store.load_summary("protocol-eight") is None
    assert "summary_generate_btn" in _keys(at)
    review_store.save_original_summary("protocol-eight", "Original text.", "llm")
    review_store.approve_summary("protocol-eight")
    at = _step2(guide=_approved_guide())
    assert "summary_generate_btn" in _keys(at)
    assert next(b for b in at.button if b.key == "summary_generate_btn").disabled
    after = review_store.load_summary("protocol-eight")
    assert is_summary_approved(after) and after["draft_text"] == "Original text."


def test_step2_generate_saves_summary_from_mock_writer():
    guide = _approved_guide()
    text = "## About the Study\n\nA short mocked summary."
    with patch("ui.summary_panel.generate_patient_summary", return_value=(text, "llm")):
        at = _step2(guide=guide)
        at = _click(at, "summary_generate_btn")
    record = review_store.load_summary("protocol-eight")
    assert record["original_text"] == text and not is_summary_approved(record)


def test_regeneration_after_ack_replaces_and_withdraws_approval():
    review_store.save_original_summary("protocol-eight", "Old text.", "llm")
    review_store.approve_summary("protocol-eight")
    with patch("ui.summary_panel.generate_patient_summary", return_value=("New text.", "llm")):
        at = _step2(guide=_approved_guide())
        at.checkbox(key="summary_regenerate_ack").check().run()
        at = _click(at, "summary_generate_btn")
    record = review_store.load_summary("protocol-eight")
    assert record["draft_text"] == "New text." and not is_summary_approved(record)


def test_prompt_requests_headings_and_short_paragraphs():
    prompt = build_summary_prompt({"Risks": ["Iron can upset the stomach."]})
    for phrase in ("## ", "3 to 5 sentences", "blank line", "Who\n  Can Participate",
                   "Possible Risks and Discomforts", "Never add a heading"):
        assert phrase in prompt


def test_extractive_fallback_is_headed_paragraphs():
    guide = _approved_guide()
    text = extractive_summary(guide)
    assert text.startswith("## ") and "\n\n" in text


def test_format_splits_long_block_without_changing_words():
    out = format_summary_markdown(LONG)
    assert "\n\n" in out
    assert out.replace("\n\n", " ") == LONG
    assert all(len(p.split(". ")) <= 4 for p in out.split("\n\n"))


def test_format_keeps_headings_lists_and_manual_breaks():
    text = "## Risks\n\nShort one.\n\n- item a\n- item b\n\nLast paragraph."
    assert format_summary_markdown(text) == text


def test_format_does_not_split_decimals():
    text = "Take 0.5 mL now. Then 5 mL later. Next 2.5 mL. Then 1.5 mL. Also 3.5 mL. Done 4.5 mL."
    assert "0.5 mL" in format_summary_markdown(text)
    assert "2.5 mL" in format_summary_markdown(text)


def test_stored_text_unchanged_by_display_formatting():
    review_store.save_original_summary("protocol-eight", LONG, "llm")
    assert review_store.load_summary("protocol-eight")["draft_text"] == LONG
    format_summary_markdown(LONG)
    assert review_store.load_summary("protocol-eight")["draft_text"] == LONG


def test_patient_portal_uses_same_formatting():
    import inspect
    from ui import patient_view
    src = inspect.getsource(patient_view)
    assert "format_summary_markdown(guide.summary)" in src
