"""Slim nav + sticky compact review-status bar with clickable filters."""

from services.guide_review import ALL, APPROVED, EDITED, PENDING, REJECTED, matches_filter
from ui import split_pane, theme
from ui.patient_guide_tab import FILTER_KEY
from tests.publish_helpers import gk
from tests.test_patient_guide_ui import (
    _approved_guide, _click, _prepare_review_app,
)


def _filter_labels(at):
    return {
        b.key: b.label
        for b in at.button
        if b.key and b.key.startswith("guide_filter_")
    }


def _count_from_label(label: str) -> int:
    return int(str(label).rsplit(" ", 1)[-1].split("/")[0].strip())


def test_bar_shows_live_counts_and_actions():
    at = _prepare_review_app(guide=_approved_guide())
    labels = _filter_labels(at)
    assert set(labels) == {
        "guide_filter_all",
        "guide_filter_approved",
        "guide_filter_pending",
        "guide_filter_rejected",
        "guide_filter_edited",
    }
    assert _count_from_label(labels["guide_filter_pending"]) == 0
    assert _count_from_label(labels["guide_filter_approved"]) == _count_from_label(
        labels["guide_filter_all"]
    ) > 0
    assert at.session_state.get(FILTER_KEY, ALL) == ALL
    keys = {b.key for b in at.button}
    assert {"guide_approve_all_btn", "publish_protocol_btn"} <= keys
    assert at.button(key="guide_approve_all_btn").disabled  # nothing pending
    assert "ctai-rbar-progress" in " ".join(
        str(getattr(e, "value", "")) for e in at.get("html")
    )


def test_counts_update_after_reject_and_bulk_approval():
    guide = _approved_guide()
    items = guide.all_items()
    items[0].approved, items[0].rejected = False, True
    at = _prepare_review_app(guide=guide)
    labels = _filter_labels(at)
    assert _count_from_label(labels["guide_filter_rejected"]) == 1
    assert _count_from_label(labels["guide_filter_approved"]) == len(items) - 1


def test_filter_buttons_show_matching_statements_and_persist():
    guide = _approved_guide()
    items = guide.all_items()
    items[0].approved, items[0].rejected = False, True
    items[1].approved = False
    items[2].generated_by = "coordinator"
    at = _prepare_review_app(guide=guide)
    total_cards = len([b for b in at.button if b.key and b.key.endswith("_approve")])

    at = _click(at, "guide_filter_rejected")
    assert at.session_state[FILTER_KEY] == REJECTED
    assert len([b for b in at.button if b.key and b.key.endswith("_reject")]) == 0
    assert any(b.key and b.key.endswith("_reconsider") for b in at.button)

    at = at.run()  # filter survives a rerun
    assert at.session_state[FILTER_KEY] == REJECTED

    at = _click(at, "guide_filter_pending")
    pending_cards = [b for b in at.button if b.key and b.key.endswith("_approve")]
    assert len(pending_cards) == 1
    assert any("No statements match" not in str(getattr(m, "value", "")) for m in at.info) or pending_cards

    at = _click(at, "guide_filter_edited")
    assert at.session_state[FILTER_KEY] == EDITED
    edited_id = items[2].item_id
    assert any(
        b.key == f"guide_{edited_id}_unapprove" or b.key == f"guide_{edited_id}_approve"
        for b in at.button
    )

    at = _click(at, "guide_filter_all")
    assert len([b for b in at.button if b.key and b.key.endswith("_approve")]) + len(
        [b for b in at.button if b.key and b.key.endswith("_unapprove")]
    ) + len([b for b in at.button if b.key and b.key.endswith("_reconsider")]) >= total_cards


def test_empty_filter_shows_a_simple_message():
    guide = _approved_guide()
    at = _prepare_review_app(guide=guide)
    at = _click(at, "guide_filter_rejected")
    assert any("No statements match this filter" in m.value for m in at.info)
    # Underlying decisions are unchanged.
    assert all(not i.rejected for i in at.session_state["patient_guide"].all_items())


def test_matches_filter_helpers():
    guide = _approved_guide()
    item = guide.all_items()[0]
    assert matches_filter(item, ALL)
    assert matches_filter(item, APPROVED)
    item.approved = False
    assert matches_filter(item, PENDING)
    item.rejected = True
    assert matches_filter(item, REJECTED)
    item.rejected = False
    item.generated_by = "coordinator"
    assert matches_filter(item, EDITED)


def test_bulk_approve_requires_confirmation_then_updates_pending():
    guide = _approved_guide()
    for item in guide.all_items():
        item.approved = False
    at = _prepare_review_app(guide=guide)
    before = _filter_labels(at)
    assert not at.button(key="guide_approve_all_btn").disabled
    at = _click(at, "guide_approve_all_btn")  # opens inline confirm
    assert "guide_approve_all_confirm_btn" in {b.key for b in at.button}
    assert any("pending" in (c.value or "").lower() for c in at.caption)
    at = _click(at, "guide_approve_all_confirm_btn")
    after = _filter_labels(at)
    assert _count_from_label(before["guide_filter_approved"]) == 0
    assert _count_from_label(after["guide_filter_approved"]) > 0
    assert _count_from_label(after["guide_filter_pending"]) == 0


def test_bulk_approve_cancel_leaves_pending_unchanged():
    guide = _approved_guide()
    for item in guide.all_items():
        item.approved = False
    at = _prepare_review_app(guide=guide)
    at = _click(at, "guide_approve_all_btn")
    at = _click(at, "guide_approve_all_cancel_btn")
    assert all(not i.approved for i in at.session_state["patient_guide"].all_items())
    assert "guide_approve_all_btn" in {b.key for b in at.button}


def test_bulk_approve_records_overrides_for_flagged_and_skips_decided():
    guide = _approved_guide()
    items = guide.all_items()
    items[0].approved = False
    items[0].source_status = "unverified"
    items[0].source_issue = "Not found."
    items[1].approved, items[1].rejected = False, True
    # items[2:] stay approved
    at = _prepare_review_app(guide=guide)
    at = _click(at, "guide_approve_all_btn")
    assert any("AI-flagged" in (c.value or "") for c in at.caption)
    at = _click(at, "guide_approve_all_confirm_btn")
    saved = at.session_state["patient_guide"].all_items()
    assert saved[0].approved and saved[0].override
    assert saved[1].rejected and not saved[1].approved
    assert all(i.approved for i in saved[2:])


def test_dashboard_side_panel_is_gone():
    at = _prepare_review_app(guide=_approved_guide())
    html = " ".join(str(getattr(e, "value", "")) for e in at.get("html"))
    assert "coordinator dashboard" not in html.lower()
    assert not hasattr(split_pane, "COORDINATOR_CONTROLS_HTML")
    assert "coord_panel" not in theme.DARK_WORKSPACE_CSS + theme.REVIEW_BAR_CSS
    assert len(at.slider) == 0


def test_sticky_css_and_responsive_rules():
    css = theme.REVIEW_BAR_CSS
    assert "position:sticky" in css and "top:54px" in css and "top:0" in css
    assert css.index("z-index:300") < css.index("z-index:290")
    assert "@media (max-width: 900px)" in css and "flex-wrap:wrap" in css
    assert css.count("{") == css.count("}")
    assert ":has(.st-key-review_bar)" in css  # only on the coordinator review page
    assert "max-height:50px" in theme.DARK_WORKSPACE_CSS or "max-height:50px" in theme.CSS
    assert "guide_filters" in css
    # Filter pills must keep full labels/counts (no ellipsis clipping).
    assert "min-width:max-content" in css and "text-overflow:clip" in css
    assert "overflow:visible" in css


def test_patient_assistant_panel_unchanged():
    assert "st-key-chat_panel" in theme.DARK_WORKSPACE_CSS
    assert "position:fixed" in theme.DARK_WORKSPACE_CSS
    assert "study assistant" in split_pane.CONTROLS_HTML
    assert "ctai.chat.frac" in split_pane._SCRIPT
