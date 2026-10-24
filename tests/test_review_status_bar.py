"""Slim nav + sticky compact review-status bar (replaces the dashboard side panel)."""

import re

from ui import split_pane, theme
from tests.test_patient_guide_ui import (
    _approved_guide, _click, _prepare_review_app,
)


def _html(at):
    return " ".join(str(getattr(e, "value", "")) for e in at.get("html"))


def _stats(at):
    found = re.findall(r'class="ctai-rstat[^"]*">(\w+)<b>([^<]+)</b>', _html(at))
    return {k: v for k, v in found}


def test_bar_shows_live_counts_and_actions():
    at = _prepare_review_app(guide=_approved_guide())
    stats = _stats(at)
    assert set(stats) == {"Total", "Approved", "Pending", "Rejected", "Edited"}
    assert stats["Pending"] == "0" and int(stats["Approved"]) == int(stats["Total"]) > 0
    keys = {b.key for b in at.button}
    assert {"guide_approve_all_btn", "publish_protocol_btn"} <= keys
    assert "ctai-rbar-progress" in _html(at) and "width:100%" in _html(at)


def test_counts_update_after_reject_and_bulk_approval():
    guide = _approved_guide()
    items = guide.all_items()
    items[0].approved, items[0].rejected = False, True
    at = _prepare_review_app(guide=guide)
    stats = _stats(at)
    assert stats["Rejected"] == "1"
    assert int(stats["Approved"]) == len(items) - 1


def test_bulk_approve_in_bar_updates_pending():
    guide = _approved_guide()
    for item in guide.all_items():
        item.approved = False
    at = _prepare_review_app(guide=guide)
    before = _stats(at)
    at = _click(at, "guide_approve_all_btn")
    after = _stats(at)
    assert before["Approved"] == "0" and int(after["Approved"]) > 0
    assert int(after["Pending"]) <= int(before["Pending"])


def test_dashboard_side_panel_is_gone():
    at = _prepare_review_app(guide=_approved_guide())
    assert "coordinator dashboard" not in _html(at).lower()
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


def test_patient_assistant_panel_unchanged():
    assert "st-key-chat_panel" in theme.DARK_WORKSPACE_CSS
    assert "position:fixed" in theme.DARK_WORKSPACE_CSS
    assert "study assistant" in split_pane.CONTROLS_HTML
    assert "ctai.chat.frac" in split_pane._SCRIPT
