"""The coordinator decides; AI verification informs and overrides are audited."""

import pytest

from agents.plain_language_agent import (
    active_overrides,
    approval_block_reason,
    approve_all_unflagged,
    approve_all_with_override,
    approve_item,
    edit_item,
    item_cleared,
    reject_item,
    reverify_item,
    verification_warnings,
)
from app import publish_block_reason
from services import review_store
from services.guide_review import (
    APPROVED,
    EDITED,
    PENDING,
    REJECTED,
    build_published_snapshot,
    decision_block_reason,
    review_state,
    state_counts,
)
from tests.publish_helpers import gk
from tests.test_patient_guide_ui import (
    _approved_guide,
    _click,
    _eight_statement_result,
    _prepare_review_app,
)
from ui.patient_guide_tab import is_fully_approved_snapshot

REASON = "Checked page 3 of the protocol myself; wording is accurate."


def _guide():
    guide = _approved_guide(_eight_statement_result())
    for item in guide.all_items():
        item.approved = False
        item.needs_review, item.review_reasons = False, []
    return guide


def _warn(item, issue="Not found on the cited page."):
    item.source_status = "unverified"
    item.source_issue = issue
    return item


# --- individual decisions ------------------------------------------------------


def test_verified_statement_is_approved_without_override():
    item = _guide().all_items()[0]
    assert not verification_warnings(item)
    approve_item(item)
    assert item.approved and not item.override
    assert review_state(item) == APPROVED


def test_failed_verification_needs_acknowledged_reason_and_records_override():
    item = _warn(_guide().all_items()[0])
    assert approval_block_reason(item) is None  # a warning is not a blocker
    assert review_state(item) == PENDING
    for missing in (None, "", "   "):
        with pytest.raises(ValueError):
            approve_item(item, missing)
        assert not item.approved
    approve_item(item, REASON)
    assert item.approved and item.override and item.override_reason == REASON
    assert any("Not verified" in w for w in item.override_warnings)
    assert item.override_at
    # The AI result is stored separately and is not rewritten as "verified".
    assert item.source_status == "unverified"
    assert item.source_issue == "Not found on the cited page."
    assert review_state(item) == APPROVED


def test_editing_an_overridden_statement_removes_the_override():
    item = _warn(_guide().all_items()[0])
    approve_item(item, REASON)
    edit_item(item, item.plain_text + " Please ask the study team.")
    assert not item.approved and not item.override and item.override_reason is None


def test_new_warning_after_override_invalidates_the_decision():
    item = _warn(_guide().all_items()[0])
    approve_item(item, REASON)
    assert item_cleared(item)
    item.source_issue = "Now contradicted by another page."
    assert not item_cleared(item)
    assert review_state(item) == PENDING


def test_unsafe_edit_cannot_be_approved_even_with_a_reason():
    item = _guide().all_items()[0]
    edit_item(item, "You should stop taking your medicine.")
    assert approval_block_reason(item)
    with pytest.raises(ValueError):
        approve_item(item, REASON)
    assert not item.approved


def test_any_statement_can_be_rejected_verified_or_not():
    guide = _guide()
    verified, flagged = guide.all_items()[0], _warn(guide.all_items()[1])
    reject_item(verified)
    reject_item(flagged)
    assert review_state(verified) == review_state(flagged) == REJECTED
    snapshot = build_published_snapshot(guide)
    assert verified.plain_text not in snapshot.model_dump_json()
    assert flagged.plain_text not in snapshot.model_dump_json()


def test_edit_then_save_check_then_approve():
    item = _warn(_guide().all_items()[0])
    page = item.original_text
    edit_item(item, page)
    assert reverify_item(item, source_page_text=page) == []
    assert not verification_warnings(item)
    approve_item(item)
    assert review_state(item) == EDITED and not item.override


# --- approve all -----------------------------------------------------------------


def test_approve_all_skips_warnings_and_keeps_existing_decisions():
    guide = _guide()
    items = guide.all_items()
    _warn(items[0])
    reject_item(items[1])
    approve_item(items[2])
    assert approve_all_unflagged(guide) == len(items) - 3
    assert not items[0].approved and not items[1].approved
    assert all(i.approved for i in items[2:])
    assert not active_overrides(guide)
    assert approve_all_unflagged(guide) == 0


def test_bulk_override_needs_a_reason_and_records_each_override():
    guide = _guide()
    items = guide.all_items()
    _warn(items[0])
    _warn(items[1], "Wording differs from the page.")
    reject_item(items[2])
    approve_all_unflagged(guide)
    with pytest.raises(ValueError):
        approve_all_with_override(guide, " ")
    assert not items[0].approved
    assert approve_all_with_override(guide, REASON) == 2
    assert {i.item_id for i in active_overrides(guide)} == {
        items[0].item_id, items[1].item_id,
    }
    assert all(i.override_reason == REASON for i in active_overrides(guide))
    assert not items[2].approved and items[2].rejected


# --- counters ---------------------------------------------------------------------


def test_counters_have_no_flagged_status_and_count_edits():
    guide = _guide()
    items = guide.all_items()
    _warn(items[0])
    approve_item(items[1])
    reject_item(items[2])
    edit_item(items[3], "In short: " + items[3].original_text)
    assert reverify_item(items[3], source_page_text=items[3].original_text) == []
    approve_item(items[3])
    counts = state_counts(guide)
    assert counts["all"] == len(items)
    assert counts[APPROVED] == 1 and counts[EDITED] == 1
    assert counts[REJECTED] == 1
    assert counts[PENDING] == len(items) - 3
    assert counts["edited_total"] == 1
    assert not any("flag" in k or "reverif" in k for k in counts)


# --- publication ------------------------------------------------------------------


def _all_approved_with_one_override():
    guide = _guide()
    approve_all_unflagged(guide)
    first = guide.all_items()[0]
    first.approved = False
    _warn(first)
    approve_item(first, REASON)
    return guide, first


def test_overrides_must_be_confirmed_before_publication_and_are_audited():
    guide, first = _all_approved_with_one_override()
    assert "confirm" in decision_block_reason(guide)
    assert not is_fully_approved_snapshot(build_published_snapshot(guide))

    guide.overrides_confirmed = [first.item_id]
    assert decision_block_reason(guide) is None
    snapshot = build_published_snapshot(guide)
    assert is_fully_approved_snapshot(snapshot)
    kept = next(i for i in snapshot.all_items() if i.item_id == first.item_id)
    # The override and the AI result travel with the publication record.
    assert kept.override and kept.override_reason == REASON
    assert kept.source_status == "unverified" and kept.source_issue

    # A further override invalidates the earlier confirmation.
    other = guide.all_items()[1]
    other.approved = False
    _warn(other)
    approve_item(other, REASON)
    assert "confirm" in decision_block_reason(guide)


def test_unconfirmed_override_snapshot_is_never_patient_valid():
    guide, _ = _all_approved_with_one_override()
    assert not is_fully_approved_snapshot(build_published_snapshot(guide))


def test_approval_flag_without_override_is_not_published():
    guide = _guide()
    approve_all_unflagged(guide)
    bad = _warn(guide.all_items()[0])
    bad.approved = True  # forced flag, no override recorded
    assert "decision" in decision_block_reason(guide)
    assert bad.plain_text not in build_published_snapshot(guide).model_dump_json()


def test_unsafe_statement_blocks_publication_despite_override():
    guide, first = _all_approved_with_one_override()
    guide.overrides_confirmed = [first.item_id]
    unsafe = guide.all_items()[1]
    edit_item(unsafe, "You should stop taking your medicine.")
    unsafe.approved = True
    assert decision_block_reason(guide)
    assert unsafe.plain_text not in build_published_snapshot(guide).model_dump_json()


def test_incomplete_generation_still_blocks_with_overrides():
    guide, first = _all_approved_with_one_override()
    guide.overrides_confirmed = [first.item_id]
    guide.generation_complete = False
    assert "still being generated" in decision_block_reason(guide)


# --- through the app (audit history, widgets, patient portal) ------------------------


def _app_with_warning():
    result = _eight_statement_result()
    guide = _approved_guide(result)
    flagged = _warn(guide.all_items()[0])
    flagged.approved = False
    return result, guide, flagged, _prepare_review_app(result, guide)


def test_override_in_the_review_page_is_acknowledged_audited_and_confirmed():
    result, guide, flagged, at = _app_with_warning()
    page = " ".join(str(e.value) for e in at.get("html"))
    assert "Flagged" not in page and "Needs reverification" not in page
    for label in ("Total", "Approved", "Pending", "Rejected", "Edited"):
        assert label in page
    key = gk(at, 0, 0, "approve")
    assert at.button(key=key).disabled
    assert at.button(key="publish_protocol_btn").disabled

    # The acknowledgement alone is not enough.
    at.checkbox(key=gk(at, 0, 0, "ack")).check()
    at = at.run()
    assert at.button(key=key).disabled
    at.text_input(key=gk(at, 0, 0, "reason")).set_value(REASON)
    at = at.run()
    assert not at.button(key=key).disabled
    at = _click(at, key)

    saved = at.session_state["patient_guide"].all_items()[0]
    assert saved.approved and saved.override and saved.override_reason == REASON
    history = review_store.history("protocol-eight")
    entry = next(h for h in history if h["action"] == "approved_with_override")
    assert REASON in entry["detail"]
    # Publication still waits for the explicit confirmation of the override.
    assert at.button(key="publish_protocol_btn").disabled
    assert "confirm" in publish_block_reason(
        at.session_state["patient_guide"], result
    ).lower()

    at.checkbox(key="guide_overrides_ack").check()
    at = at.run()
    at = _click(at, "guide_overrides_confirm_btn")
    assert "overrides_confirmed" in [
        h["action"] for h in review_store.history("protocol-eight")
    ]
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")

    published = review_store.load_publications()["protocol-eight"]["guide"]
    item = next(i for i in published.all_items() if i.item_id == saved.item_id)
    assert item.override_reason == REASON and item.source_status == "unverified"
    assert is_fully_approved_snapshot(published)


def test_rejecting_a_flagged_statement_publishes_without_it():
    result, guide, flagged, at = _app_with_warning()
    at = _click(at, gk(at, 0, 0, "reject"))
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")
    published = review_store.load_publications()["protocol-eight"]["guide"]
    assert flagged.plain_text not in published.model_dump_json()
    assert flagged.original_text not in published.model_dump_json()


def test_bulk_buttons_separate_verified_from_warned_statements():
    result = _eight_statement_result()
    guide = _guide()
    for item in guide.all_items()[:2]:
        _warn(item)
    at = _prepare_review_app(result, guide)
    at = _click(at, "guide_approve_all_btn")
    saved = at.session_state["patient_guide"]
    assert [i.approved for i in saved.all_items()] == [False, False] + [True] * (
        len(saved.all_items()) - 2
    )
    bulk = "guide_bulk_override_btn"
    assert at.button(key=bulk).disabled
    at.checkbox(key="guide_bulk_override_ack").check()
    at = at.run()
    assert at.button(key=bulk).disabled  # a reason is also required
    at.text_input(key="guide_bulk_override_reason").set_value(REASON)
    at = at.run()
    at = _click(at, bulk)
    saved = at.session_state["patient_guide"]
    assert all(i.approved for i in saved.all_items())
    assert len(active_overrides(saved)) == 2
    assert "approved_all_with_override" in [
        h["action"] for h in review_store.history("protocol-eight")
    ]
    assert at.button(key="publish_protocol_btn").disabled  # overrides unconfirmed


def test_flagged_rewrite_fallback_needs_the_bulk_override_to_be_approved_in_bulk():
    guide = _guide()
    flagged = guide.all_items()[0]
    flagged.needs_review = True
    flagged.review_reasons = ["These numbers from the original are missing: 45"]
    assert approve_all_unflagged(guide) == len(guide.all_items()) - 1
    assert not flagged.approved
    approve_all_with_override(guide, REASON)
    assert flagged.approved and flagged.override and flagged.override_reason == REASON
