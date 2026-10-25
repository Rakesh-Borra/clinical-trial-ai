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
    assert {"guide_filter_all", "guide_filter_approved", "guide_filter_pending"} <= keys
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
    approve_item(item)  # Approve records the override; wording stays unverified
    assert item.approved and item.override and item.source_status == "unverified"


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
    at = _review_and_approve(at)
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


# --- Approve Summary sits below the collapsed section ----------------------

def _review_and_approve(at):
    at.checkbox(key=_reviewed_key(at)).check()
    return _click(at.run(), "summary_approve_btn")


def _reviewed_key(at):
    return next(c.key for c in at.checkbox if (c.key or "").startswith("summary_reviewed_"))


def _unapproved_app():
    at = _app()
    review_store.withdraw_summary_approval("protocol-eight")
    return at.run()


def test_approve_button_is_outside_the_collapsed_expander_and_not_duplicated():
    at = _unapproved_app()
    summary = next(e for e in at.expander if e.label == "Patient Summary")
    assert not summary.proto.expanded
    assert [b.key for b in at.button].count("summary_approve_btn") == 1
    assert "summary_approve_btn" not in [b.key for b in summary.button]
    assert "summary_approve_btn" not in [b.key for b in summary.children.values() if hasattr(b, "key")]


def test_approval_is_blocked_until_the_summary_has_been_reviewed():
    at = _unapproved_app()
    assert at.button(key="summary_approve_btn").disabled
    assert at.button(key="publish_protocol_btn").disabled
    at.checkbox(key=_reviewed_key(at)).check()
    at = at.run()
    assert not at.button(key="summary_approve_btn").disabled
    assert at.button(key="publish_protocol_btn").disabled  # still needs the approval itself


def test_approving_after_review_shows_status_and_hides_the_button():
    at = _review_and_approve(_unapproved_app())
    assert "summary_approve_btn" not in [b.key for b in at.button]
    assert any(s.value == "Summary Approved" for s in at.success)
    assert not at.button(key="publish_protocol_btn").disabled


def test_editing_an_approved_summary_requires_review_and_approval_again():
    at = _review_and_approve(_unapproved_app())
    old_key = [c.key for c in at.checkbox if (c.key or "").startswith("summary_reviewed_")]
    at.text_area(key=_summary_key()).set_value(GENERIC_SUMMARY + " Questions are welcome.")
    at = _click(at.run(), "summary_save_btn")
    assert not any(s.value == "Summary Approved" for s in at.success)
    assert at.button(key="summary_approve_btn").disabled  # the earlier tick does not carry over
    assert at.button(key="publish_protocol_btn").disabled
    assert old_key == [] or _reviewed_key(at) not in old_key
    at = _review_and_approve(at)
    assert not at.button(key="publish_protocol_btn").disabled


# --- final approval validation: stale override confirmation -----------------

def _guide_with_stale_override_confirmation():
    from agents.plain_language_agent import approve_item

    # Must match the eight-statement pipeline used by _app() / _result().
    guide = _approved_guide(_result())
    item = guide.all_items()[0]
    item.source_status = "unverified"
    approve_item(item, "checked the protocol page")
    guide.overrides_confirmed = [item.item_id]
    item.source_status = "verified"  # later re-verified: the override no longer exists
    approve_item(item)
    return guide, item


def test_stale_override_confirmation_does_not_block_a_fully_approved_guide():
    from services.guide_review import build_published_snapshot
    from ui.patient_guide_tab import final_validation_reason

    guide, item = _guide_with_stale_override_confirmation()
    assert guide.overrides_confirmed == [item.item_id]
    snapshot = build_published_snapshot(guide)
    assert final_validation_reason(snapshot) is None
    assert snapshot.overrides_confirmed == []


def test_publish_is_enabled_with_stale_override_confirmation_and_acknowledged_gaps():
    guide, _ = _guide_with_stale_override_confirmation()
    at = _app(guide=guide)
    assert not at.button(key="publish_protocol_btn").disabled
    assert not any("final approval validation" in e.value for e in list(at.error) + list(at.warning))


def test_recorded_override_publishes_without_a_second_confirmation_list():
    from agents.plain_language_agent import approve_item

    guide = _approved_guide()
    item = guide.all_items()[0]
    item.source_status = "unverified"
    approve_item(item)
    guide.overrides_confirmed = []  # Approve is enough; list is not required
    assert gr_reason(guide) is None


def test_forced_approval_without_override_still_blocks_publication():
    guide = _approved_guide()
    item = guide.all_items()[0]
    item.source_status = "unverified"
    item.approved = True  # no override recorded
    assert gr_reason(guide)


def gr_reason(guide):
    from services.guide_review import decision_block_reason

    return decision_block_reason(guide)


def test_final_validation_names_the_failing_requirement():
    from services.guide_review import build_published_snapshot
    from ui.patient_guide_tab import final_validation_reason

    guide = _approved_guide()
    snapshot = build_published_snapshot(guide)
    snapshot.all_items()[0].analysis_complete = False
    assert "analysis verification" in final_validation_reason(snapshot)
    snapshot = build_published_snapshot(guide)
    snapshot.generation_complete = False
    assert "generation" in final_validation_reason(snapshot)
    snapshot = build_published_snapshot(guide)
    snapshot.all_items()[0].approved = False
    assert "not approved" in final_validation_reason(snapshot)


def test_final_validation_accepts_rehydrated_guide_from_dump_or_foreign_class():
    """
    Session-held guides can fail isinstance(PatientGuide) after a module reload
    or when restored as a dict. Re-validation must keep real approvals intact.
    """
    from models.plain_language_models import PatientGuide, as_patient_guide
    from services.guide_review import build_published_snapshot
    from ui.patient_guide_tab import final_validation_reason

    guide = _approved_guide(_result())
    assert final_validation_reason(guide.model_dump()) is None
    assert final_validation_reason(guide.model_dump_json()) == "The patient guide could not be read."

    foreign = as_patient_guide(guide.model_dump())
    assert isinstance(foreign, PatientGuide)
    # Simulate a stale class identity: dump/validate still yields a readable guide.
    snapshot = build_published_snapshot(foreign)
    assert final_validation_reason(snapshot) is None
    assert len(snapshot.all_items()) == len(guide.all_items())

    at = _app(guide=guide)
    at.session_state["patient_guide"] = guide.model_dump()
    at = at.run()
    assert isinstance(at.session_state["patient_guide"], PatientGuide)
    assert not at.button(key="publish_protocol_btn").disabled
    assert not any("could not be read" in (w.value or "") for w in at.warning)


def test_acknowledged_missing_categories_do_not_trigger_final_validation_error():
    from app import publish_block_reason  # noqa: F401  (import check)

    at = _app()
    for key in ("eligibility", "interventions", "visits", "procedures", "risks", "consent"):
        review_store.clear_coverage_override("protocol-eight", key)
    at = at.run()
    reason = " ".join(m.value for m in list(at.warning) + list(at.error))
    assert "final approval validation" not in reason
