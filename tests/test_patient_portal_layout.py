import inspect
import re

from streamlit.testing.v1 import AppTest

from agents.guide_patient_agent import APPROVED_STATUS
from agents.plain_language_agent import approve_item
from tests.test_guide_patient_agent import _guide
from tests.test_patient_guide_ui import APP_PATH, _approved_guide, _as
from ui import landing, patient_view, split_pane, theme


def _fully_approved_t2d_guide():
    guide = _guide()
    for item in guide.all_items():
        approve_item(item)
    return guide


def _patient_app(protocols=None, protocol_id="p1"):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["published_protocols"] = protocols or {
        "p1": {"name": "Diabetes Study", "guide": _approved_guide()}
    }
    at = _as(at, "patient").run()
    at.selectbox(key="patient_protocol_select").set_value(protocol_id)
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


def test_patient_chat_opens_without_streamlit_gap_crash():
    at = _patient_app()
    assert not at.exception
    assert "patient_chat_input" in {w.key for w in at.chat_input}
    assert "new_chat_btn" in {b.key for b in at.button}
    err = " ".join(str(getattr(e, "message", e)) for e in at.exception)
    assert "StreamlitInvalidColumnGapError" not in err
    assert "xxsmall" not in err and "xsmall" not in err


def test_chat_toolbar_uses_supported_streamlit_15_args():
    source = inspect.getsource(patient_view._render_chat_toolbar)
    assert 'gap="xxsmall"' not in source
    assert 'gap="xsmall"' not in source
    assert "gap=None" in source
    popover_call = source.split("st.popover(", 1)[1].split("):", 1)[0]
    assert "type=" not in popover_call
    assert "key=" not in popover_call


def test_t2_diabetes_chat_uses_approved_grounded_answer_and_safety():
    guide = _fully_approved_t2d_guide()
    at = _patient_app(
        {"t2d": {"name": "Type 2 Diabetes Treatment Study", "guide": guide}},
        protocol_id="t2d",
    )
    assert not at.exception
    assert "patient_chat_input" in {w.key for w in at.chat_input}

    at.chat_input(key="patient_chat_input").set_value(
        "Do I need to have Type 2 diabetes to join this study?"
    )
    at = at.run()
    assert not at.exception
    answer = at.session_state["patient_history"][-1]["answer"]
    assert "Type 2 diabetes" in answer.answer_text
    assert answer.citations
    assert all(c.verification_status == APPROVED_STATUS for c in answer.citations)
    assert "protocol page" in answer.answer_text.lower()

    # Fresh chat so prior eligibility context cannot skew safety classification.
    at.button(key="new_chat_btn").click()
    at = at.run()
    at.chat_input(key="patient_chat_input").set_value(
        "Can I stop taking my medication?"
    )
    at = at.run()
    safety = at.session_state["patient_history"][-1]["answer"]
    assert safety.intent == "medical_advice_refused"
    assert not safety.citations
    lower = safety.answer_text.lower()
    assert "study team" in lower or "clinician" in lower or "medical advice" in lower


def test_no_unsupported_streamlit_gap_or_popover_kwargs_in_ui():
    """Fresh-clone compatibility: Streamlit 1.50 rejects these arguments."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "ui"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r'gap\s*=\s*["\']xx?small["\']', text):
            offenders.append(f"{path.name}: unsupported gap")
        if re.search(r'gap\s*=\s*["\']none["\']', text):
            # Error message mentions "none", but only Python None is accepted.
            offenders.append(f"{path.name}: gap='none' string is invalid; use None")
        if re.search(r"\.popover\([^)]*\btype\s*=", text, re.S):
            offenders.append(f"{path.name}: popover type=")
        if re.search(r"\.popover\([^)]*\bkey\s*=", text, re.S):
            offenders.append(f"{path.name}: popover key=")
    assert offenders == []
    # Landing must keep a key-free Account popover after the Streamlit 1.50 fix.
    assert "landing_account" not in inspect.getsource(landing.render_landing)
