"""Focused tests for top-bar Back to Top / Go to Bottom chrome."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from tests.auth_helpers import sign_in_as
from tests.test_patient_guide_ui import APP_PATH
from ui.landing import (
    COORDINATOR,
    PATIENT,
    VIEW_HOME,
    VIEW_LIST,
    VIEW_VERSIONS,
    VIEW_WORKFLOW,
)
from ui.scroll_nav import (
    SCROLL_ACTIONS_HTML,
    pick_scroll_target,
    scroll_target_candidates,
    should_show_scroll_nav,
)


def _html_blob(at: AppTest) -> str:
    return " ".join(h.value for h in at.get("html"))


def test_hidden_on_landing_and_login() -> None:
    assert should_show_scroll_nav(role=None, on_landing=True) is False
    assert should_show_scroll_nav(role=None, on_landing=False) is False


def test_hidden_on_coordinator_dashboard_home() -> None:
    assert (
        should_show_scroll_nav(role=COORDINATOR, view=VIEW_HOME, on_landing=False)
        is False
    )
    assert should_show_scroll_nav(role=COORDINATOR, view=None, on_landing=False) is False


def test_shown_on_long_coordinator_pages() -> None:
    for view in (VIEW_LIST, VIEW_VERSIONS, VIEW_WORKFLOW):
        assert (
            should_show_scroll_nav(role=COORDINATOR, view=view, on_landing=False)
            is True
        )


def test_shown_on_patient_portal_and_profile() -> None:
    assert should_show_scroll_nav(role=PATIENT, on_landing=False) is True
    assert (
        should_show_scroll_nav(
            role=COORDINATOR,
            view=VIEW_HOME,
            show_profile=True,
            on_landing=False,
        )
        is True
    )


def test_scroll_actions_html_is_main_document_markup_not_iframe() -> None:
    html = SCROLL_ACTIONS_HTML
    assert "ctai-scroll-actions" in html
    assert 'data-ctai-scroll="top"' in html
    assert 'data-ctai-scroll="bottom"' in html
    assert "↑" in html and "↓" in html
    assert "iframe" not in html.lower()
    assert "window.parent" not in html
    assert "components.html" not in html


def test_render_topbar_scroll_buttons_uses_st_html(monkeypatch) -> None:
    import ui.scroll_nav as scroll_nav

    calls = []

    def fake_html(body, **kwargs):
        calls.append({"body": body, "kwargs": kwargs})

    monkeypatch.setattr(scroll_nav.st, "html", fake_html)
    scroll_nav.render_topbar_scroll_buttons()

    assert len(calls) == 2
    assert "ctai-scroll-btn" in calls[0]["body"]
    assert "<script>" in calls[1]["body"]
    assert calls[1]["kwargs"].get("unsafe_allow_javascript") is True
    assert "window.parent" not in calls[1]["body"]
    assert "components" not in calls[1]["body"]


def test_script_binds_clicks_and_targets_st_main() -> None:
    import ui.scroll_nav as scroll_nav

    script = scroll_nav._SCRIPT
    assert '[data-testid="stMain"]' in script
    assert "findScrollContainer" in script
    assert "addEventListener(\"click\"" in script or "addEventListener('click'" in script
    assert "bindButtons" in script
    assert "MutationObserver" in script
    assert "behavior: \"smooth\"" in script or "behavior:'smooth'" in script
    # Must not stop after first mount without rebinding replacement buttons.
    assert 'dataset.ctaiScrollBound' in script or "dataset.ctaiScrollBound" in script
    assert scroll_target_candidates()[0] == '[data-testid="stMain"]'


def test_pick_scroll_target_prefers_st_main_over_window_like_nodes() -> None:
    chosen = pick_scroll_target(
        [
            {
                "id": "html",
                "overflow_y": "visible",
                "scroll_height": 800,
                "client_height": 800,
            },
            {
                "id": "stAppViewContainer",
                "overflow_y": "hidden",
                "scroll_height": 800,
                "client_height": 800,
            },
            {
                "id": "stMain",
                "overflow_y": "auto",
                "scroll_height": 2400,
                "client_height": 800,
            },
        ]
    )
    assert chosen is not None
    assert chosen["id"] == "stMain"


def test_pick_scroll_target_ignores_non_scrollable_overflow() -> None:
    assert (
        pick_scroll_target(
            [
                {
                    "id": "hidden",
                    "overflow_y": "hidden",
                    "scroll_height": 2000,
                    "client_height": 400,
                },
                {
                    "id": "tiny",
                    "overflow_y": "auto",
                    "scroll_height": 405,
                    "client_height": 400,
                },
            ]
        )
        is None
    )


def test_render_top_bar_includes_scroll_markup_in_nav_path(monkeypatch) -> None:
    """Buttons must be mounted from render_top_bar, not only a detached helper."""
    import ui.landing as landing
    import ui.scroll_nav as scroll_nav

    recorded = {"html": [], "weights": None, "nested_weights": None}

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def button(self, *args, **kwargs):
            return False

        def popover(self, *args, **kwargs):
            return _Col()

        def caption(self, *args, **kwargs):
            return None

        def columns(self, weights, **kwargs):
            recorded["nested_weights"] = list(weights)
            return [_Col(), _Col()]

    def fake_columns(weights, **kwargs):
        recorded["weights"] = list(weights)
        return [_Col() for _ in weights]

    class _Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_html(body, **kwargs):
        recorded["html"].append(body)

    monkeypatch.setattr(landing.st, "columns", fake_columns)
    monkeypatch.setattr(landing.st, "container", lambda **kwargs: _Container())
    monkeypatch.setattr(
        landing.st,
        "session_state",
        {landing.COORD_VIEW_KEY: VIEW_WORKFLOW},
    )
    monkeypatch.setattr(landing, "_ensure_page_history", lambda: None)
    monkeypatch.setattr(landing, "_render_history_arrows", lambda *a, **k: None)
    monkeypatch.setattr(
        landing.authz,
        "current_user",
        lambda: {"full_name": "Ada", "role": COORDINATOR},
    )
    monkeypatch.setattr(scroll_nav.st, "html", fake_html)

    landing.render_top_bar(COORDINATOR)

    joined = "\n".join(recorded["html"])
    assert "ctai-scroll-btn" in joined
    assert 'data-ctai-scroll="top"' in joined
    assert 'data-ctai-scroll="bottom"' in joined
    assert recorded["weights"] == [2, 0.5, 0.5, 5.2, 2.4]
    assert recorded["nested_weights"] == [1.05, 1.55]


def test_render_top_bar_omits_scroll_on_coordinator_home(monkeypatch) -> None:
    import ui.landing as landing
    import ui.scroll_nav as scroll_nav

    recorded = {"html": [], "weights": None}

    class _Col:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def button(self, *args, **kwargs):
            return False

        def popover(self, *args, **kwargs):
            return _Col()

        def caption(self, *args, **kwargs):
            return None

    def fake_columns(weights, **kwargs):
        recorded["weights"] = list(weights)
        return [_Col() for _ in weights]

    class _Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(landing.st, "columns", fake_columns)
    monkeypatch.setattr(landing.st, "container", lambda **kwargs: _Container())
    monkeypatch.setattr(
        landing.st,
        "session_state",
        {landing.COORD_VIEW_KEY: VIEW_HOME},
    )
    monkeypatch.setattr(landing, "_ensure_page_history", lambda: None)
    monkeypatch.setattr(landing, "_render_history_arrows", lambda *a, **k: None)
    monkeypatch.setattr(
        landing.authz,
        "current_user",
        lambda: {"full_name": "Ada", "role": COORDINATOR},
    )
    monkeypatch.setattr(
        scroll_nav.st,
        "html",
        lambda *a, **k: recorded["html"].append(a[0] if a else ""),
    )

    landing.render_top_bar(COORDINATOR)

    assert recorded["html"] == []
    assert recorded["weights"] == [2, 0.5, 0.5, 6, 1.3]


def test_apptest_workflow_topbar_includes_scroll_buttons() -> None:
    """End-to-end: workflow top bar HTML contains ↑/↓ in the navigation path."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator", coord_view=VIEW_WORKFLOW)
    at.session_state["coord_step"] = "1 · Upload"
    at.session_state["coord_max_step"] = 0
    at = at.run()

    assert not at.exception
    html = _html_blob(at)
    assert "ctai-scroll-actions" in html
    assert 'data-ctai-scroll="top"' in html
    assert 'data-ctai-scroll="bottom"' in html
    assert "↑" in html and "↓" in html


def test_apptest_home_and_login_omit_scroll_buttons() -> None:
    login = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert "ctai-scroll-btn" not in _html_blob(login)

    home = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(home.session_state, "coordinator", coord_view=VIEW_HOME)
    home = home.run()
    assert not home.exception
    assert "ctai-scroll-btn" not in _html_blob(home)
