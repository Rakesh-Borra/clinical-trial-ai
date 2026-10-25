"""Simplified Review page: statement decisions, manual statements, patient summary, publish."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from agents.plain_language_agent import approve_item
from services import manual_statements as ms
from services import review_store
from services.guide_review import PENDING, build_published_snapshot, review_state
from tests.auth_helpers import sign_in_as
from tests.publish_helpers import GENERIC_SUMMARY, gk
from tests.test_patient_guide_ui import (
    APP_PATH,
    _approved_guide,
    _click,
    _eight_statement_result,
    _prepare_review_app,
)

PAGES = {
    "1": "Participants must be between 18 and 65 years old. Participants must give written informed consent.",
    "2": "Follow-up visits happen at Week 2 and Week 4.",
    "3": "Blood collection takes 10 mL at four visits.",
}
SUPPORTED = "Participants must give written informed consent."
UNSUPPORTED = "Participants receive a free gym membership."


def _result(with_pages=True):
    result = _eight_statement_result()
    if with_pages:
        result["source_pages"] = dict(PAGES)
    return result


def _app(with_pages=True, guide=None):
    result = _result(with_pages)
    return _prepare_review_app(result, guide or _approved_guide(result))


def _add(at, category, text, page, eligibility=None):
    at = _click(at, "add_stmt_open_btn")
    at.selectbox(key="add_stmt_category").select(category)
    at = at.run()
    if eligibility:
        at.radio(key="add_stmt_eligibility").set_value(eligibility)
    at.text_area(key="add_stmt_text").set_value(text)
    at.number_input(key="add_stmt_page").set_value(page)
    return _click(at, "add_stmt_save_btn")


def _manual(at):
    return [i for i in at.session_state["patient_guide"].all_items() if i.added_by_coordinator]


def _blank_guide():
    return _approved_guide(_result())


def _summary_key():
    record = review_store.load_summary("protocol-eight")
    return f"summary_text_protocol-eight_{record['updated_at']}"


# --- layout ----------------------------------------------------------------

def test_review_page_has_four_sections_and_no_clutter():
    at = _app()
    keys = {b.key for b in at.button}
    assert {"add_stmt_open_btn", "publish_protocol_btn"} <= keys
    assert not [k for k in keys if k and k.startswith("guide_filter_")]
    summary = [e for e in at.expander if e.label == "Patient Summary"]
    assert len(summary) == 1 and summary[0].proto.expanded is False
    text = " ".join(str(getattr(e, "value", "")) for e in list(at.markdown) + list(at.subheader))
    assert "Required topic" not in text and "Completeness" not in text


# --- statement decisions ---------------------------------------------------

def test_approve_reject_edit_and_cancel_edit():
    result = _result()
    guide = _approved_guide(result)
    guide.sections()[0].items[0].approved = False
    at = _prepare_review_app(result, guide)
    first = at.session_state["patient_guide"].sections()[0].items[0]

    at = _click(at, gk(at, 0, 0, "approve"))
    assert at.session_state["patient_guide"].sections()[0].items[0].approved
    assert review_store.load_draft("protocol-eight")["guide"].sections()[0].items[0].approved

    at = _click(at, gk(at, 2, 0, "reject"))
    rejected = at.session_state["patient_guide"].sections()[2].items[0]
    assert rejected.rejected and not rejected.approved
    snapshot = build_published_snapshot(at.session_state["patient_guide"])
    assert rejected.item_id not in [i.item_id for i in snapshot.all_items()]

    key = gk(at, 0, 0, "text")
    original = at.session_state["patient_guide"].sections()[0].items[0].plain_text
    at.text_area(key=key).set_value("Something I will cancel")
    at = at.run()
    at = _click(at, gk(at, 0, 0, "cancel"))
    assert at.text_area(key=key).value == original
    assert at.session_state["patient_guide"].sections()[0].items[0].plain_text == original

    at.text_area(key=key).set_value(original + " Updated.")
    at = at.run()
    at = _click(at, gk(at, 0, 0, "save"))
    edited = at.session_state["patient_guide"].sections()[0].items[0]
    assert edited.plain_text.endswith("Updated.") and edited.generated_by == "coordinator"
    assert not edited.approved and edited.original_text == first.original_text


# --- manual statements -----------------------------------------------------

def test_supported_manual_statement_is_added_pending_and_checked_against_its_page():
    at = _add(_app(), "Consent and withdrawal", SUPPORTED, 1)
    (item,) = _manual(at)
    assert review_state(item) == PENDING
    assert item.source_status == "verified" and item.source_page == 1
    assert item.section == "consent_withdrawal"
    assert item.item_id and item.generated_by == "coordinator"
    assert not item.approved


def test_unsupported_manual_statement_is_never_marked_verified():
    at = _add(_app(), "Other", UNSUPPORTED, 1)
    (item,) = _manual(at)
    assert item.source_status == "unverified" and item.needs_review
    assert item.verification != "verified"
    with pytest.raises(ValueError):  # approving needs a recorded override
        approve_item(item)


def test_manual_statement_without_pdf_text_says_verification_unavailable():
    at = _add(_app(with_pages=False), "Other", SUPPORTED, 4)
    (item,) = _manual(at)
    assert item.source_status == "unverified"
    assert "unavailable" in item.source_issue.lower()


def test_page_number_alone_is_not_evidence():
    item = ms.add_manual_statement(
        _blank_guide(), "Study visits and schedule", "Visits happen daily.", 2, PAGES
    )
    assert item.source_status != "verified"


@pytest.mark.parametrize(
    "text,page,message",
    [("   ", 1, "empty"), (SUPPORTED, 0, "page number"), (SUPPORTED, 99, "not in this protocol")],
)
def test_manual_statement_validation(text, page, message):
    with pytest.raises(ms.ManualStatementError, match=message):
        ms.add_manual_statement(_blank_guide(), "Other", text, page, PAGES)


def test_duplicate_manual_statement_is_rejected():
    guide = _blank_guide()
    ms.add_manual_statement(guide, "Other", SUPPORTED, 1, PAGES)
    with pytest.raises(ms.ManualStatementError, match="already added"):
        ms.add_manual_statement(guide, "Other", "  " + SUPPORTED.upper() + " ", 1, PAGES)


def test_form_shows_validation_error_and_adds_nothing():
    at = _add(_app(), "Other", "   ", 1)
    assert any("cannot be empty" in e.value for e in at.error)
    assert not _manual(at)


def test_cancel_closes_the_form():
    at = _click(_app(), "add_stmt_open_btn")
    at = _click(at, "add_stmt_cancel_btn")
    assert any(b.key == "add_stmt_open_btn" for b in at.button)
    assert not _manual(at)


def test_eligibility_choice_and_category_sections():
    guide = _blank_guide()
    cannot = ms.add_manual_statement(guide, "Eligibility", SUPPORTED, 1, PAGES, "Who cannot join")
    assert cannot.section == "exclusion_criteria"
    blood = ms.add_manual_statement(
        guide, "Procedures and blood collection", "Blood collection takes 10 mL at four visits.", 3, PAGES
    )
    assert blood.section == "procedures"
    risk = ms.add_manual_statement(guide, "Risks and safety", "Tell the team about problems.", 1, PAGES)
    assert risk.section == "safety_information"


def test_manual_statement_persists_and_stays_in_its_own_protocol():
    at = _add(_app(), "Consent and withdrawal", SUPPORTED, 1)
    (item,) = _manual(at)
    saved = review_store.load_draft("protocol-eight")["guide"]
    assert [i.item_id for i in saved.all_items() if i.added_by_coordinator] == [item.item_id]

    review_store.save_draft("another-protocol", _blank_guide(), name="Another")
    other = review_store.load_draft("another-protocol")["guide"]
    assert not [i for i in other.all_items() if i.added_by_coordinator]


def test_manual_statement_needs_approval_before_it_is_published():
    at = _add(_app(), "Consent and withdrawal", SUPPORTED, 1)
    (item,) = _manual(at)
    snapshot = build_published_snapshot(at.session_state["patient_guide"])
    assert item.plain_text not in [i.plain_text for i in snapshot.all_items()]
    assert at.button(key="publish_protocol_btn").disabled
    at = _click(at, f"guide_{item.item_id}_approve")
    (approved,) = _manual(at)
    assert approved.approved
    snapshot = build_published_snapshot(at.session_state["patient_guide"])
    assert approved.plain_text in [i.plain_text for i in snapshot.all_items()]


# --- patient summary -------------------------------------------------------

def test_summary_is_collapsed_editable_and_original_is_preserved():
    at = _app()
    assert next(e for e in at.expander if e.label == "Patient Summary").proto.expanded is False
    key = _summary_key()
    assert at.text_area(key=key).value == GENERIC_SUMMARY
    at.text_area(key=key).set_value(GENERIC_SUMMARY + " Edited by the coordinator.")
    at = _click(at.run(), "summary_save_btn")
    saved = review_store.load_summary("protocol-eight")
    assert saved["draft_text"].endswith("Edited by the coordinator.")
    assert saved["original_text"] == GENERIC_SUMMARY
    assert not saved.get("approved_at")  # editing withdrew the approval
    assert any("Summary saved" in s.value for s in at.success)
    assert review_store.summary_audit("protocol-eight")


def test_cancelling_summary_edits_restores_the_saved_text():
    at = _app()
    key = _summary_key()
    at.text_area(key=key).set_value("Draft I do not want")
    at = _click(at.run(), "summary_cancel_btn")
    assert at.text_area(key=key).value == GENERIC_SUMMARY
    assert review_store.load_summary("protocol-eight")["draft_text"] == GENERIC_SUMMARY


def test_unsaved_summary_edit_blocks_publish_with_a_reason():
    at = _app()
    assert not at.button(key="publish_protocol_btn").disabled
    at.text_area(key=_summary_key()).set_value(GENERIC_SUMMARY + " Unsaved.")
    at = at.run()
    assert at.button(key="publish_protocol_btn").disabled
    assert any("Save or cancel your edits" in w.value for w in at.warning)


def test_saved_edit_must_be_reapproved_then_publishes_the_approved_summary():
    at = _app()
    new_text = GENERIC_SUMMARY + " Ask the study team if you have questions."
    at.text_area(key=_summary_key()).set_value(new_text)
    at = _click(at.run(), "summary_save_btn")
    assert at.button(key="publish_protocol_btn").disabled  # approval invalidated
    at = _click(at, "summary_approve_btn")
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")
    published = review_store.load_publications()["protocol-eight"]["guide"]
    assert published.summary.strip() == new_text.strip()


def test_rejected_statements_never_reach_the_published_guide():
    at = _app()
    at = _click(at, gk(at, 0, 1, "reject"))
    rejected = at.session_state["patient_guide"].sections()[0].items[1]
    at = _click(at, "publish_protocol_btn")
    published = review_store.load_publications()["protocol-eight"]["guide"].model_dump_json()
    assert rejected.plain_text not in published and rejected.original_text not in published


# --- authorization ---------------------------------------------------------

def test_patients_do_not_get_review_controls():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at = at.run()
    keys = {b.key for b in at.button}
    assert not keys & {"add_stmt_open_btn", "add_stmt_save_btn", "publish_protocol_btn", "summary_save_btn"}


def _patient_writes():
    import streamlit as st

    from services import review_store as rs

    try:
        rs.save_summary_draft("p", "text")
        st.write("allowed")
    except Exception:
        st.write("denied")


def test_review_writes_are_denied_to_patients_server_side():
    at = AppTest.from_function(_patient_writes, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at = at.run()
    assert [m.value for m in at.markdown] == ["denied"]
