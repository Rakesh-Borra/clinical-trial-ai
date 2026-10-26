"""Regression tests for Streamlit-compatible parent-document JS injection."""

from __future__ import annotations

import ui.app_js as app_js
import ui.scroll_nav as scroll_nav
import ui.split_pane as split_pane


def test_run_in_app_document_wraps_script_for_parent(monkeypatch) -> None:
    captured = []

    def fake_html(html, height=None, scrolling=False, **kwargs):
        captured.append(
            {"html": html, "height": height, "scrolling": scrolling, "kwargs": kwargs}
        )

    monkeypatch.setattr(app_js.components, "html", fake_html)
    app_js.run_in_app_document("window.__ctaiProbe = true;")

    assert len(captured) == 1
    call = captured[0]
    assert call["height"] == 0
    assert call["scrolling"] is False
    assert "unsafe_allow_javascript" not in call["kwargs"]
    assert "window.parent" in call["html"]
    assert "window.__ctaiProbe = true;" in call["html"]
    assert call["html"].strip().startswith("<script>")


def test_scroll_nav_and_split_pane_use_app_js(monkeypatch) -> None:
    scripts = []
    monkeypatch.setattr(scroll_nav.st, "html", lambda *a, **k: None)
    monkeypatch.setattr(scroll_nav, "run_in_app_document", scripts.append)
    monkeypatch.setattr(split_pane, "run_in_app_document", scripts.append)

    scroll_nav.render_topbar_scroll_buttons()
    split_pane.render_script()

    assert scripts == [scroll_nav._SCRIPT, split_pane._SCRIPT]
