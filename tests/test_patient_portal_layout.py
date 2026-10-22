from streamlit.testing.v1 import AppTest

from tests.test_patient_guide_ui import APP_PATH, _approved_guide, _as
from ui import patient_view, split_pane, theme


def _patient_app(protocols=None):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["published_protocols"] = protocols or {
        "p1": {"name": "Diabetes Study", "guide": _approved_guide()}
    }
    at = _as(at, "patient").run()
    at.selectbox(key="patient_protocol_select").set_value("p1")
    at = at.run()
    at.button(key="view_protocol_summary_btn").click()
    return at.run()


def test_no_slider_or_columns_and_panels_exist():
    at = _patient_app()
    assert not at.select_slider
    assert not hasattr(patient_view, "CHAT_SIZES")
    assert not hasattr(patient_view, "GUIDE_CHAT_COLUMNS")
    assert not at.columns or all(
        "chat" not in str(getattr(c, "key", "")) for c in at.columns
    )


def test_chat_history_survives_rerun_and_keeps_disclaimer():
    at = _patient_app()
    at.button(key="suggest_0").click()
    at = at.run()
    assert len(at.session_state["patient_history"]) == 1
    at = at.run()
    assert len(at.session_state["patient_history"]) == 1
    assert any("not medical advice" in str(c.value).lower() for c in at.caption)


def test_chat_history_not_shared_between_protocols():
    at = _patient_app(
        {
            "p1": {"name": "Diabetes Study", "guide": _approved_guide()},
            "p2": {"name": "Second Study", "guide": _approved_guide()},
        }
    )
    at.button(key="suggest_0").click()
    at = at.run()
    assert at.session_state["patient_history"]
    at.selectbox(key="patient_protocol_select").set_value("p2")
    at = at.run()
    assert at.session_state["patient_history"] == []


def test_fixed_split_pane_css_is_present():
    css = theme.CSS + theme.DARK_WORKSPACE_CSS
    assert "position:fixed" in css
    assert ".ctai-divider" in css and 'data-ctai-chat="min"' in css
    assert ".st-key-chat_messages" in css and "overflow-y:auto" in css
    assert "position:sticky" in css  # topbar only
    assert "@media (max-width: 900px)" in css


def test_split_pane_script_and_controls():
    import re

    # Streamlit's sanitizer drops script bodies containing markup-like text.
    assert not re.search(r"<[/\w!]", split_pane._SCRIPT)
    for action in ("min", "max"):
        assert f'data-ctai-action="{action}"' in split_pane.CONTROLS_HTML
    assert "ctai.chat.frac" in split_pane._SCRIPT


def test_assistant_panel_renders_controls_and_script():
    at = _patient_app()
    html = " ".join(str(m.value) for m in at.markdown)
    assert "ctai-ctl-min" in html
