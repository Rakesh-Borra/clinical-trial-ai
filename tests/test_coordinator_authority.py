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


def test_failed_verification_approve_records_override_without_a_written_reason():
    item = _warn(_guide().all_items()[0])
    assert approval_block_reason(item) is None  # a warning is not a blocker
    assert review_state(item) == PENDING
    # Approve itself is the explicit human override; no written reason required.
    approve_item(item)
    assert item.approved and item.override and item.override_reason is None
    assert any("Not verified" in w for w in item.override_warnings)
    assert item.override_at
    # The AI result is stored separately and is not rewritten as "verified".
    assert item.source_status == "unverified"
    assert item.source_issue == "Not found on the cited page."
    assert review_state(item) == APPROVED


def test_optional_override_reason_is_kept_when_supplied():
    item = _warn(_guide().all_items()[0])
    approve_item(item, REASON)
    assert item.override and item.override_reason == REASON


def test_editing_an_overridden_statement_removes_the_override():
    item = _warn(_guide().all_items()[0])
    approve_item(item)
    edit_item(item, item.plain_text + " Please ask the study team.")
    assert not item.approved and not item.override and item.override_reason is None


def test_new_warning_after_override_invalidates_the_decision():
    item = _warn(_guide().all_items()[0])
    approve_item(item)
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


def test_bulk_override_records_each_override_without_requiring_a_reason():
    guide = _guide()
    items = guide.all_items()
    _warn(items[0])
    _warn(items[1], "Wording differs from the page.")
    reject_item(items[2])
    approve_all_unflagged(guide)
    assert approve_all_with_override(guide) == 2
    assert {i.item_id for i in active_overrides(guide)} == {
        items[0].item_id, items[1].item_id,
    }
    assert all(i.override and i.override_at for i in active_overrides(guide))
    assert {i.item_id for i in active_overrides(guide)} <= set(guide.overrides_confirmed)
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
    approve_item(first)
    guide.overrides_confirmed = sorted(set(guide.overrides_confirmed) | {first.item_id})
    return guide, first


def test_approve_on_flagged_statement_is_enough_to_publish():
    guide, first = _all_approved_with_one_override()
    assert decision_block_reason(guide) is None
    snapshot = build_published_snapshot(guide)
    assert is_fully_approved_snapshot(snapshot)
    kept = next(i for i in snapshot.all_items() if i.item_id == first.item_id)
    # The override and the AI result travel with the publication record.
    assert kept.override and kept.override_at
    assert kept.override_reason is None  # never fabricated
    assert kept.source_status == "unverified" and kept.source_issue

    # Another flagged approval is also enough; no second confirmation step.
    other = guide.all_items()[1]
    other.approved = False
    _warn(other)
    approve_item(other)
    guide.overrides_confirmed = sorted(set(guide.overrides_confirmed) | {other.item_id})
    assert decision_block_reason(guide) is None


def test_approval_flag_without_override_is_not_published():
    guide = _guide()
    approve_all_unflagged(guide)
    bad = _warn(guide.all_items()[0])
    bad.approved = True  # forced flag, no override recorded
    assert decision_block_reason(guide)
    snapshot = build_published_snapshot(guide)
    assert bad.plain_text not in snapshot.model_dump_json()
    assert bad.item_id not in [i.item_id for i in snapshot.all_items()]


def test_stale_override_without_timestamp_is_not_published():
    guide, first = _all_approved_with_one_override()
    first.override_at = None  # incomplete / stale recorded override
    # Treated as pending until Approve records a complete override again.
    assert "decision" in (decision_block_reason(guide) or "").lower()
    assert first.plain_text not in build_published_snapshot(guide).model_dump_json()


def test_unsafe_statement_blocks_publication_despite_override():
    guide, first = _all_approved_with_one_override()
    unsafe = guide.all_items()[1]
    edit_item(unsafe, "You should stop taking your medicine.")
    unsafe.approved = True
    assert decision_block_reason(guide)
    assert unsafe.plain_text not in build_published_snapshot(guide).model_dump_json()


def test_incomplete_generation_still_blocks_with_overrides():
    guide, _ = _all_approved_with_one_override()
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
    assert "AI flagged this statement" in page
    assert "warning(s)" not in page and "Needs reverification" not in page
    assert "Not verified against" not in page and "AI note:" not in page
    assert "Closest lines" not in page and "Check result:" not in page
    assert "Short reason for the audit trail" not in " ".join(
        str(getattr(t, "label", "")) for t in at.text_input
    )
    filter_labels = " ".join(
        b.label for b in at.button if b.key and b.key.startswith("guide_filter_")
    )
    for label in ("Total", "Approved", "Pending", "Rejected", "Edited"):
        assert label in filter_labels
    key = gk(at, 0, 0, "approve")
    assert not at.button(key=key).disabled  # warning is advisory; Approve is enabled
    assert at.button(key="publish_protocol_btn").disabled

    at = _click(at, key)

    saved = at.session_state["patient_guide"].all_items()[0]
    assert saved.approved and saved.override and saved.override_at
    assert saved.override_reason is None  # never fabricated
    assert saved.item_id in at.session_state["patient_guide"].overrides_confirmed
    history = review_store.history("protocol-eight")
    entry = next(h for h in history if h["action"] == "approved_with_override")
    detail = entry["detail"]
    assert saved.item_id in detail and "flagged_approved" in detail
    assert "actor=" in detail and saved.override_at in detail
    # Approve is the confirmation — publish is unlocked without a second step.
    at = at.run()
    assert decision_block_reason(at.session_state["patient_guide"]) is None
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")

    published = review_store.load_publications()["protocol-eight"]["guide"]
    item = next(i for i in published.all_items() if i.item_id == saved.item_id)
    assert item.override and item.source_status == "unverified"
    assert is_fully_approved_snapshot(published)


def test_existing_valid_overrides_publish_without_reinventing_confirmation():
    """Previously approved flagged statements with a recorded override can publish."""
    result = _eight_statement_result()
    guide = _approved_guide(result)
    flagged = _warn(guide.all_items()[0])
    flagged.override = True
    flagged.override_warnings = verification_warnings(flagged)
    flagged.override_at = "2026-01-01T00:00:00+00:00"
    flagged.override_reason = None
    # Deliberately omit overrides_confirmed — must not invent it, but publish
    # still succeeds because the item-level override is valid.
    guide.overrides_confirmed = []
    assert decision_block_reason(guide) is None
    at = _prepare_review_app(result, guide)
    assert not at.button(key="publish_protocol_btn").disabled
    at = _click(at, "publish_protocol_btn")
    published = review_store.load_publications()["protocol-eight"]["guide"]
    kept = next(i for i in published.all_items() if i.item_id == flagged.item_id)
    assert kept.override and kept.override_at


def test_verified_statement_card_shows_no_ai_flag_warning():
    at = _prepare_review_app(_eight_statement_result(), _approved_guide(_eight_statement_result()))
    page = " ".join(str(e.value) for e in at.get("html"))
    assert "AI flagged this statement" not in page


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
    # Inline confirmation required before any pending statement is approved.
    assert all(not i.approved for i in at.session_state["patient_guide"].all_items()[:2])
    assert any("AI-flagged" in (c.value or "") for c in at.caption)
    at = _click(at, "guide_approve_all_confirm_btn")
    saved = at.session_state["patient_guide"]
    assert all(i.approved for i in saved.all_items())
    assert len(active_overrides(saved)) == 2
    assert "approved_all_with_override" in [
        h["action"] for h in review_store.history("protocol-eight")
    ]
    assert not at.button(key="publish_protocol_btn").disabled


def test_flagged_rewrite_fallback_needs_the_bulk_override_to_be_approved_in_bulk():
    guide = _guide()
    flagged = guide.all_items()[0]
    flagged.needs_review = True
    flagged.review_reasons = ["These numbers from the original are missing: 45"]
    assert approve_all_unflagged(guide) == len(guide.all_items()) - 1
    assert not flagged.approved
    approve_all_with_override(guide, REASON)
    assert flagged.approved and flagged.override and flagged.override_reason == REASON
