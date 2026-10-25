"""Missing critical categories warn and need an acknowledgment instead of blocking outright."""

from __future__ import annotations

import pytest

import app as app_module
from services import review_store
from tests.publish_helpers import GENERIC_SUMMARY
from tests.test_patient_guide_ui import (
    _approved_guide,
    _click,
    _eight_statement_result,
    _prepare_review_app,
)

PID = "protocol-eight"
ACK = "publish_ack_checkbox"


def _incomplete_app():
    result = _eight_statement_result()
    at = _prepare_review_app(result, _approved_guide(result))
    for key in ("eligibility", "visits", "procedures", "interventions", "risks", "consent"):
        review_store.clear_coverage_override(PID, key)
    review_store.save_draft(PID, at.session_state["patient_guide"], name="Eight Statement Study")
    return at.run()


def _history(action):
    return [h for h in review_store.history(PID, 50) if h["action"] == action]


def _tick(at, value=True):
    at.checkbox(key=ACK).set_value(value)
    return at.run()


def _context(at):
    return app_module.acknowledgment_context(
        at.session_state["patient_guide"], at.session_state["pipeline_result"], PID
    )


def test_complete_protocol_needs_no_acknowledgment():
    result = _eight_statement_result()
    at = _prepare_review_app(result, _approved_guide(result))
    assert not at.checkbox
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")
    assert review_store.publication_status(PID) == "published"
    assert not _history("published_with_missing_categories")


def test_missing_categories_show_a_warning_and_block_until_acknowledged():
    at = _incomplete_app()
    missing, _ = _context(at)
    assert missing
    warning = " ".join(w.value for w in at.warning)
    assert all(label in warning for label in missing)
    assert at.checkbox(key=ACK).label.startswith("I have reviewed the missing categories")
    assert at.button(key="publish_protocol_btn").disabled
    assert review_store.publication_status(PID) is None


def test_acknowledging_enables_publish_and_writes_the_audit_trail():
    at = _incomplete_app()
    missing, _ = _context(at)
    at = _tick(at)
    assert not at.button(key="publish_protocol_btn").disabled
    ack = _history("incomplete_publication_acknowledged")
    assert len(ack) == 1
    at = _click(at, "publish_protocol_btn")
    assert review_store.publication_status(PID) == "published"
    (entry,) = _history("published_with_missing_categories")
    detail = entry["detail"]
    assert "coordinator@example.test" in detail or "@" in detail
    assert PID in detail and "version 1.0" in detail and "acknowledged 20" in detail
    assert all(label in detail for label in missing)


def test_unticking_withdraws_the_acknowledgment():
    at = _tick(_incomplete_app())
    at = _tick(at, False)
    assert at.button(key="publish_protocol_btn").disabled
    assert review_store.load_incomplete_acknowledgment(PID) is None


def test_changing_statements_resets_the_acknowledgment():
    at = _tick(_incomplete_app())
    assert at.checkbox(key=ACK).value
    guide = at.session_state["patient_guide"]
    guide.all_items()[0].approved = False
    at.session_state["patient_guide"] = guide
    at = at.run()
    assert not at.checkbox(key=ACK).value
    assert at.button(key="publish_protocol_btn").disabled


def test_changing_the_summary_resets_the_acknowledgment():
    at = _tick(_incomplete_app())
    review_store.save_summary_draft(PID, GENERIC_SUMMARY + " Changed.")
    review_store.approve_summary(PID)
    at = at.run()
    assert not at.checkbox(key=ACK).value
    assert at.button(key="publish_protocol_btn").disabled


def test_changing_the_version_resets_the_acknowledgment():
    at = _incomplete_app()
    _, before = _context(at)
    with review_store.closing(review_store._connect()) as conn, conn:
        conn.execute("UPDATE protocol_versions SET version_label = '9.9' WHERE protocol_id = ?", (PID,))
    _, after = _context(at)
    assert before != after


def test_server_side_publish_refuses_without_a_valid_acknowledgment():
    at = _incomplete_app()
    missing, fingerprint = _context(at)
    from services.guide_review import build_published_snapshot

    snapshot = build_published_snapshot(at.session_state["patient_guide"])
    with pytest.raises(review_store.AcknowledgmentError):
        review_store.publish(PID, "N", snapshot, missing, fingerprint)  # no stored ack
    with pytest.raises(review_store.AcknowledgmentError):
        review_store.publish(PID, "N", snapshot, missing, None)
    review_store.record_incomplete_acknowledgment(PID, "stale-fingerprint", missing, "1.0")
    with pytest.raises(review_store.AcknowledgmentError):
        review_store.publish(PID, "N", snapshot, missing, fingerprint)  # content changed
    assert review_store.publication_status(PID) is None


def test_the_ui_button_path_also_refuses_a_stale_acknowledgment():
    at = _tick(_incomplete_app())
    review_store.record_incomplete_acknowledgment(PID, "stale", _context(at)[0], "1.0")
    at = at.run()
    assert at.button(key="publish_protocol_btn").disabled


def test_other_safeguards_still_block_when_acknowledged():
    at = _tick(_incomplete_app())
    guide = at.session_state["patient_guide"]
    guide.all_items()[0].approved = False
    guide.all_items()[0].rejected = False
    at.session_state["patient_guide"] = guide
    at = at.run()
    assert at.button(key="publish_protocol_btn").disabled
    assert any("decision" in w.value.lower() for w in at.warning)


def test_unapproved_summary_still_blocks_when_acknowledged():
    at = _tick(_incomplete_app())
    review_store.withdraw_summary_approval(PID)
    at = at.run()
    assert at.button(key="publish_protocol_btn").disabled
