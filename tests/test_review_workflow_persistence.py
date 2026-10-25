"""Coordinator edit/save/approve/publish/unpublish workflow and persistence."""

from streamlit.testing.v1 import AppTest

from agents.plain_language_agent import (
    approve_item,
    edit_item,
    reject_item,
    reverify_item,
)
from app import publish_block_reason
from services import review_store
from tests.publish_helpers import gk
from services.guide_review import PENDING, build_published_snapshot, review_state
from tests.test_patient_guide_ui import (
    APP_PATH,
    _approved_guide,
    _as,
    _click,
    _eight_statement_result,
    _generated,
    _prepare_review_app,
)

NEW_TEXT = "Adults between 18 and 65 years old can be considered."


def _first(guide):
    return guide.all_items()[0]


def _fresh_guide():
    """A generated guide where nothing is approved yet."""
    guide = _approved_guide(_eight_statement_result())
    for item in guide.all_items():
        item.approved = False
    return guide


def _edit_first(at, text):
    at.text_area(key=gk(at, 0, 0, "text")).set_value(text)
    at = at.run()
    return _click(at, gk(at, 0, 0, "save"))


# --- editing and saving ----------------------------------------------------


def test_edit_updates_statement_immediately_and_persists():
    result = _eight_statement_result()
    guide = _fresh_guide()
    original_page = _first(guide).source_page
    original_text = _first(guide).original_text
    at = _prepare_review_app(result, guide)
    at = _edit_first(at, NEW_TEXT)

    item = _first(at.session_state["patient_guide"])
    assert item.plain_text == NEW_TEXT
    assert review_state(item) == PENDING
    assert item.source_page == original_page and item.original_text == original_text
    assert any("Edit saved" in w.value for w in list(at.success) + list(at.warning))
    assert at.text_area(key=gk(at, 0, 0, "text")).value == NEW_TEXT

    stored = review_store.load_latest_draft()["guide"]
    assert _first(stored).plain_text == NEW_TEXT
    assert _first(stored).source_page == original_page
    actions = [h["action"] for h in review_store.history("protocol-eight")]
    assert "edited" in actions


def test_unsaved_edit_cannot_be_approved():
    at = _prepare_review_app(_eight_statement_result(), _fresh_guide())
    at.text_area(key=gk(at, 0, 0, "text")).set_value(NEW_TEXT)
    at = at.run()
    assert at.button(key=gk(at, 0, 0, "approve")).disabled
    assert not _first(at.session_state["patient_guide"]).approved


def test_edit_is_rechecked_on_save_then_needs_explicit_approval():
    at = _prepare_review_app(_eight_statement_result(), _fresh_guide())
    at = _edit_first(at, NEW_TEXT)

    item = _first(at.session_state["patient_guide"])
    assert item.verification == "verified" and not item.approved
    assert not at.button(key=gk(at, 0, 0, "approve")).disabled

    at = _click(at, gk(at, 0, 0, "approve"))
    assert _first(at.session_state["patient_guide"]).approved
    assert _first(review_store.load_latest_draft()["guide"]).approved


def test_edit_that_adds_unsupported_number_fails_reverification():
    guide = _fresh_guide()
    item = _first(guide)
    edit_item(item, item.plain_text + " Most people are 99 years old.")
    problems = reverify_item(item)
    assert any("99" in p for p in problems)
    assert item.verification == "needs_reverification"


def test_edited_statement_cannot_bypass_verification():
    """Forced approval without Approve must not publish; Approve records an override."""
    result = _eight_statement_result()
    guide = _approved_guide(result)
    edit_item(_first(guide), NEW_TEXT)
    assert _first(guide).verification == "needs_reverification"
    _first(guide).approved = True  # forced, bypassing Approve / override recording
    assert publish_block_reason(guide, result) is not None

    # The real Approve path is allowed: it records a coordinator override.
    edit_item(_first(guide), NEW_TEXT)
    approve_item(_first(guide))
    assert _first(guide).approved and _first(guide).override
    assert "Edited wording has not been reverified" in " ".join(
        _first(guide).override_warnings or []
    )


# --- publication -----------------------------------------------------------


def _published_app():
    result = _eight_statement_result()
    at = _prepare_review_app(result, _approved_guide(result))
    return _click(at, "publish_protocol_btn")


def _patient_session():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    return _as(at, "patient").run()


def test_publish_persists_and_survives_restart():
    at = _published_app()
    assert "protocol-eight" in at.session_state["published_protocols"]
    assert review_store.publication_status("protocol-eight") == "published"

    restarted = _patient_session()  # new session, nothing in memory
    assert "protocol-eight" in restarted.session_state["published_protocols"]


def test_rejected_and_unapproved_never_published():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    rejected_text = _first(guide).plain_text
    reject_item(_first(guide))
    at = _prepare_review_app(result, guide)
    at = _click(at, "publish_protocol_btn")
    stored = review_store.load_publications()["protocol-eight"]["guide"]
    assert rejected_text not in stored.model_dump_json()

    pending = _fresh_guide()
    at = _prepare_review_app(result, pending)
    assert at.button(key="publish_protocol_btn").disabled
    assert review_store.publication_status("protocol-eight") == "published"
    assert len(stored.all_items()) == 7


def test_unpublish_withdraws_from_patients_and_assistant():
    at = _published_app()
    at = _click(at, "unpublish_protocol_btn")
    assert at.session_state["published_protocols"] == {}
    assert review_store.publication_status("protocol-eight") == "withdrawn"
    assert "no longer visible" in " ".join(s.value for s in at.success)

    restarted = _patient_session()
    assert restarted.session_state["published_protocols"] == {}


def test_withdrawal_in_another_session_reaches_open_patient_session():
    at = _published_app()
    patient = _patient_session()
    assert "protocol-eight" in patient.session_state["published_protocols"]
    _click(at, "unpublish_protocol_btn")
    patient = patient.run()
    assert patient.session_state["published_protocols"] == {}


def test_republish_replaces_record_without_duplicates():
    at = _published_app()
    at = _click(at, "publish_protocol_btn")
    assert len(review_store.load_publications()) == 1
    assert len(review_store.history("protocol-eight", 50)) >= 2


def test_published_snapshot_contains_only_approved_statements():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    hidden = guide.all_items()[-1]
    hidden.approved = False
    snapshot = build_published_snapshot(guide)
    texts = [i.plain_text for i in snapshot.all_items()]
    assert guide.all_items()[0].plain_text in texts
    assert hidden.plain_text not in texts


def test_patient_side_uses_latest_published_version():
    at = _published_app()
    at = _edit_first(at, NEW_TEXT)
    # Until the edit is approved and republished, patients keep
    # the earlier published wording.
    published = review_store.load_publications()["protocol-eight"]["guide"]
    assert NEW_TEXT not in published.model_dump_json()

    at = _click(at, gk(at, 0, 0, "approve"))
    at = _click(at, "publish_protocol_btn")
    published = review_store.load_publications()["protocol-eight"]["guide"]
    assert NEW_TEXT in published.model_dump_json()

    patient = _patient_session()
    seen = patient.session_state["published_protocols"]["protocol-eight"]["guide"]
    assert NEW_TEXT in seen.model_dump_json()


# --- restart restores the working review ----------------------------------


def test_review_edits_survive_restart():
    result = _eight_statement_result()
    at = _prepare_review_app(result, _fresh_guide())
    at = _edit_first(at, NEW_TEXT)

    fresh = AppTest.from_file(APP_PATH, default_timeout=30)
    fresh = _as(fresh, "coordinator")
    fresh.session_state["coord_view"] = "home"
    fresh = _click(fresh.run(), "home_view_btn")
    assert fresh.session_state.get("patient_guide") is None
    fresh = _click(fresh, "open_protocol_protocol-eight")
    guide = fresh.session_state["patient_guide"]
    assert _first(guide).plain_text == NEW_TEXT
    assert review_state(_first(guide)) == PENDING
    assert fresh.session_state["coordinator_protocol_id"] == "protocol-eight"
    assert fresh.text_area(key=gk(fresh, 0, 0, "text")).value == NEW_TEXT


def test_stale_text_box_is_not_reapplied_after_regeneration():
    at = _prepare_review_app(_eight_statement_result(), _fresh_guide())
    key = gk(at, 0, 0, "text")
    at.text_area(key=key).set_value("Old unsaved wording")
    at = at.run()
    at = _click(at, "guide_regenerate_btn")
    at = _generated(at)
    assert key not in at.session_state or at.session_state[key] != "Old unsaved wording"


# --- store ----------------------------------------------------------------


def test_store_upserts_never_duplicate(tmp_path):
    guide = _fresh_guide()
    for _ in range(3):
        review_store.save_draft("p1", guide, name="One")
        review_store.publish("p1", "One", guide)
    assert len(review_store.load_publications()) == 1
    assert review_store.draft_names() == {"p1": "One"}
    assert review_store.unpublish("p1") is True
    assert review_store.unpublish("p1") is False
