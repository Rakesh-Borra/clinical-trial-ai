"""Coordinator home dashboard, saved-protocol list and protocol switching."""

from __future__ import annotations

import copy

import pytest
from streamlit.testing.v1 import AppTest

from services import review_store
from tests.auth_helpers import sign_in_as
from tests.test_patient_guide_ui import (
    APP_PATH,
    _approved_guide,
    _click,
    _eight_statement_result,
    _make_account,
    _ui_login,
)
from ui import coordinator_home

NO_PROTOCOLS = "No protocols have been added yet."


def _texts(at):
    chunks = [m.value for m in at.markdown]
    # Streamlit 1.50 AppTest may expose st.html as UnknownElement without .value.
    for h in at.get("html"):
        value = getattr(h, "value", None)
        if value is not None:
            chunks.append(str(value))
    chunks.extend(c.value for c in at.caption)
    chunks.extend(i.value for i in at.info)
    chunks.extend(w.value for w in at.warning)
    return " ".join(chunks)


def _save(protocol_id, name, pdf, *, marker, with_pipeline=True, approved=True):
    """Store a saved protocol whose statements all start with `marker`."""
    result = copy.deepcopy(_eight_statement_result())
    guide = _approved_guide(result)
    for item in guide.all_items():
        item.plain_text = f"{marker}: {item.plain_text}"
        item.approved = approved
    review_store.save_draft(
        protocol_id,
        guide,
        name=name,
        uploaded_name=pdf,
        upload_fingerprint=f"fp-{protocol_id}",
        pipeline=result if with_pipeline else None,
    )
    return guide


def _dashboard(role="coordinator"):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, role, coord_view="home")
    return at.run()


def _keys(at):
    return {b.key for b in at.button}


# --- login and dashboard ---------------------------------------------------


def test_coordinator_login_opens_home_dashboard_without_cached_work():
    _save("p-old", "Earlier Study", "earlier.pdf", marker="OLD")
    address = _make_account("coordinator")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    at = _ui_login(at, address)

    assert at.session_state["role"] == "coordinator"
    assert at.session_state["coord_view"] == "home"
    assert {"home_add_btn", "home_view_btn"} <= _keys(at)
    assert not at.file_uploader
    assert at.session_state["pipeline_result"] is None
    assert at.session_state["coordinator_protocol_id"] is None
    assert "Earlier Study" not in _texts(at)
    assert "continue_summary_btn" not in _keys(at)


def test_dashboard_shows_both_options_and_no_review_forms():
    at = _dashboard()

    assert not at.exception
    assert "Add New Protocol" in _texts(at) and "View Existing Protocols" in _texts(at)
    assert not at.file_uploader and not at.text_input


def test_logout_and_login_return_to_dashboard():
    address = _make_account("coordinator")
    at = _dashboard()
    at = _click(at, "home_add_btn")
    assert at.session_state["coord_view"] == "workflow"

    at = _click(at, "logout_btn")
    assert at.session_state["role"] is None
    at = _ui_login(at, address)
    assert at.session_state["coord_view"] == "home"
    assert "home_add_btn" in _keys(at)


# --- add new protocol ------------------------------------------------------


def test_add_new_protocol_opens_clean_existing_workflow():
    _save("p-old", "Earlier Study", "earlier.pdf", marker="OLD")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator", coord_view="home")
    # Leftover temporary state from earlier work in this browser session.
    at.session_state["pipeline_result"] = _eight_statement_result()
    at.session_state["coordinator_protocol_id"] = "p-old"
    at.session_state["uploaded_name"] = "earlier.pdf"
    at.session_state["patient_guide"] = _approved_guide()
    at = at.run()
    at = _click(at, "home_add_btn")

    assert at.session_state["coord_view"] == "workflow"
    assert at.session_state["coord_step"] == "1 · Upload"
    assert at.session_state["pipeline_result"] is None
    assert at.session_state["coordinator_protocol_id"] is None
    assert at.session_state["uploaded_name"] is None
    assert at.session_state.get("patient_guide") is None
    assert at.file_uploader
    assert "workflow_back_btn" in _keys(at)
    # Saved data is untouched.
    assert [p["protocol_id"] for p in review_store.list_protocols()] == ["p-old"]


def test_back_to_dashboard_from_workflow_and_list():
    at = _click(_dashboard(), "home_add_btn")
    at = _click(at, "workflow_back_btn")
    assert at.session_state["coord_view"] == "home"
    assert "home_add_btn" in _keys(at)

    at = _click(at, "home_view_btn")
    assert at.session_state["coord_view"] == "list"
    at = _click(at, "list_back_btn")
    assert at.session_state["coord_view"] == "home"


# --- existing protocols ----------------------------------------------------


def test_empty_list_says_no_protocols_and_offers_add():
    at = _click(_dashboard(), "home_view_btn")

    assert NO_PROTOCOLS in _texts(at)
    at = _click(at, "list_add_btn")
    assert at.session_state["coord_view"] == "workflow"


def test_list_shows_only_real_saved_records_with_status():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    _save("p-two", "Anemia Study", "anemia.pdf", marker="TWO", approved=False)
    at = _click(_dashboard(), "home_view_btn")

    assert {"open_protocol_p-one", "open_protocol_p-two"} <= _keys(at)
    text = _texts(at)
    assert "Diabetes Study" in text and "diabetes.pdf" in text
    assert "Anemia Study" in text and "anemia.pdf" in text
    assert "8 of 8 statements approved" in text
    assert "0 of 8 statements approved" in text
    assert "Analysis saved" in text
    assert NO_PROTOCOLS not in text


def test_session_only_protocol_is_not_listed():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator", coord_view="home")
    at.session_state["coordinator_protocols"] = {"never-saved": {"name": "Ghost"}}
    at = _click(at.run(), "home_view_btn")

    assert NO_PROTOCOLS in _texts(at)
    assert "Ghost" not in _texts(at)


def test_opening_a_protocol_loads_its_saved_data_without_analysis(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "services.protocol_pipeline.analyze_protocol_pdf",
        lambda *a, **k: calls.append(a),
    )
    saved = _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    at = _click(_dashboard(), "home_view_btn")
    at = _click(at, "open_protocol_p-one")

    assert at.session_state["coord_view"] == "workflow"
    assert at.session_state["coord_step"] == "3 · Review"
    assert at.session_state["coordinator_protocol_id"] == "p-one"
    assert at.session_state["uploaded_name"] == "diabetes.pdf"
    assert at.session_state["coordinator_protocols"]["p-one"]["name"] == "Diabetes Study"
    loaded = at.session_state["patient_guide"]
    assert [i.plain_text for i in loaded.all_items()] == [
        i.plain_text for i in saved.all_items()
    ]
    assert at.session_state["pipeline_result"]["analysis"] is not None
    assert not calls and not at.exception
    assert "workflow_back_btn" in _keys(at)


def test_switching_protocols_never_mixes_data():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    _save("p-two", "Anemia Study", "anemia.pdf", marker="TWO")
    at = _click(_dashboard(), "home_view_btn")
    at = _click(at, "open_protocol_p-one")
    assert all(
        i.plain_text.startswith("ONE:")
        for i in at.session_state["patient_guide"].all_items()
    )

    at = _click(at, "workflow_back_btn")
    at = _click(at, "home_view_btn")
    at = _click(at, "open_protocol_p-two")

    assert at.session_state["coordinator_protocol_id"] == "p-two"
    assert at.session_state["uploaded_name"] == "anemia.pdf"
    assert all(
        i.plain_text.startswith("TWO:")
        for i in at.session_state["patient_guide"].all_items()
    )
    assert "ONE:" not in _texts(at)


def test_saved_protocol_without_analysis_is_not_faked():
    _save("p-legacy", "Legacy Study", "legacy.pdf", marker="LEG", with_pipeline=False)
    at = _click(_dashboard(), "home_view_btn")
    assert "Analysis not saved" in _texts(at)
    at = _click(at, "open_protocol_p-legacy")

    assert at.session_state["pipeline_result"] is None
    assert at.session_state.get("patient_guide") is None
    assert "was not saved" in _texts(at)
    stored = review_store.load_draft("p-legacy")
    assert stored["name"] == "Legacy Study" and stored["guide"] is not None


def test_new_upload_after_legacy_open_creates_new_protocol_and_keeps_record():
    _save("p-legacy", "Legacy Study", "legacy.pdf", marker="LEG", with_pipeline=False)
    at = _click(_dashboard(), "home_view_btn")
    at = _click(at, "open_protocol_p-legacy")
    at.file_uploader[0].set_value(("new.pdf", b"%PDF-1.4 brand new", "application/pdf"))
    at = at.run()

    assert at.session_state["coordinator_protocol_id"] not in (None, "p-legacy")
    assert "was not saved" not in _texts(at)
    assert review_store.load_draft("p-legacy")["name"] == "Legacy Study"


# --- storage helpers -------------------------------------------------------


def test_list_protocols_reports_publication_and_summary_status():
    guide = _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    review_store.save_original_summary("p-one", "Summary text.", "llm")
    review_store.approve_summary("p-one")
    review_store.publish("p-one", "Diabetes Study", guide)

    (entry,) = review_store.list_protocols()
    assert entry["publication"] == review_store.PUBLISHED
    assert entry["summary_approved"] is True
    assert coordinator_home.review_status(entry) == ("Published to patients", "teal")

    review_store.unpublish("p-one")
    (entry,) = review_store.list_protocols()
    text, _ = coordinator_home.review_status(entry)
    assert text.startswith("Withdrawn from patients") and "summary approved" in text


def test_editing_an_approved_summary_shows_summary_not_approved():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    review_store.save_original_summary("p-one", "Summary text.", "llm")
    review_store.approve_summary("p-one")
    review_store.save_summary_draft("p-one", "Edited summary text.")

    (entry,) = review_store.list_protocols()
    assert "summary not approved" in coordinator_home.review_status(entry)[0]


def test_unreadable_saved_review_is_listed_but_cannot_be_opened(tmp_path):
    import sqlite3

    _save("p-ok", "Good Study", "good.pdf", marker="OK")
    with sqlite3.connect(review_store.db_path()) as conn:
        conn.execute(
            "INSERT INTO drafts (protocol_id, name, guide_json, updated_at) "
            "VALUES ('p-bad', 'Broken Study', '{not json', '2026-01-01T00:00:00+00:00')"
        )
    entries = {e["protocol_id"]: e for e in review_store.list_protocols()}
    assert entries["p-bad"]["guide"] is None
    assert coordinator_home.review_status(entries["p-bad"])[0] == "Saved review unreadable"
    at = _click(_dashboard(), "home_view_btn")
    assert next(b for b in at.button if b.key == "open_protocol_p-bad").disabled
    assert not next(b for b in at.button if b.key == "open_protocol_p-ok").disabled


def test_format_saved_time():
    assert coordinator_home.format_saved_time("2026-10-24T20:01:05+00:00") == "2026-10-24 20:01 UTC"
    assert coordinator_home.format_saved_time("garbage") is None
    assert coordinator_home.format_saved_time(None) is None


# --- access control --------------------------------------------------------


def test_patient_never_sees_coordinator_dashboard_or_unpublished_protocols():
    _save("p-one", "Secret Draft Study", "secret.pdf", marker="ONE")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at.session_state["coord_view"] = "list"
    at = at.run()

    assert at.session_state["role"] == "patient"
    assert not at.file_uploader
    assert not {"home_add_btn", "home_view_btn", "open_protocol_p-one"} & _keys(at)
    assert "Secret Draft Study" not in _texts(at)


def test_patient_cannot_force_the_coordinator_role_to_reach_the_dashboard():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at.session_state["role"] = "coordinator"
    at.session_state["coord_view"] = "workflow"
    at = at.run()

    assert at.session_state["role"] is None
    assert not at.file_uploader


def test_signed_out_user_sees_neither_dashboard_nor_protocols():
    _save("p-one", "Secret Draft Study", "secret.pdf", marker="ONE")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["role"] = "coordinator"
    at.session_state["coord_view"] = "list"
    at = at.run()

    assert at.session_state["role"] is None
    assert "Secret Draft Study" not in _texts(at)


def test_store_listing_is_coordinator_only_inside_a_session():
    from services import authz

    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at = at.run()
    assert not at.exception
    # The same guard the app uses: patients are refused before any read.
    with pytest.raises(authz.AuthorizationError):
        _run_as_patient(review_store.list_protocols)


def _run_as_patient(func):
    """Call `func` inside a live script run with only a patient signed in."""
    from unittest.mock import patch

    with patch("streamlit.runtime.scriptrunner.get_script_run_ctx", return_value=object()), patch(
        "services.authz.current_user", return_value={"role": "patient"}
    ):
        return func()


# --- unsaved work and active analysis ---------------------------------------

import threading
from unittest.mock import patch

from services import analysis_jobs
from tests.test_patient_guide_ui import (
    _GatedAnalyzer,
    _generated,
    _pipeline_result,
    _upload,
)


@pytest.fixture(autouse=True)
def _clean_jobs():
    analysis_jobs.jobs.reset()
    yield
    analysis_jobs.jobs.reset()


def _start_running(name="Slow Study", content=b"%PDF-1.4 slow"):
    """Coordinator with an upload whose analysis is blocked mid-run."""
    analyzer = _GatedAnalyzer()
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator", coord_view="home")
    at = _click(at.run(), "home_add_btn")
    at = _upload(at, content)
    protocol_id = at.session_state["coordinator_protocol_id"]
    at.text_input(key=f"protocol_name_{protocol_id}").set_value(name)
    at = _click(at, "save_protocol_name_btn")
    patcher = patch("services.protocol_pipeline.analyze_protocol_pdf", analyzer)
    patcher.start()
    at = _click(at, "analyze_btn")
    assert analyzer.started.wait(5)
    job_id = at.session_state["analysis_job_id"]
    return at, analyzer, patcher, protocol_id, job_id


def _stop(analyzer, patcher):
    analyzer.release.set()
    patcher.stop()


def _go_dashboard(at):
    return _click(at, "workflow_back_btn")


def test_back_to_dashboard_keeps_running_analysis_and_offers_return():
    at, analyzer, patcher, protocol_id, job_id = _start_running()
    try:
        at = _go_dashboard(at)
        assert at.session_state["coord_view"] == "home"
        assert "home_resume_btn" in _keys(at)
        assert analysis_jobs.jobs.get(job_id).active
        at = _click(at, "home_resume_btn")
        assert at.session_state["coord_view"] == "workflow"
        assert at.session_state["analysis_job_id"] == job_id
    finally:
        _stop(analyzer, patcher)


def test_add_new_while_analysis_running_asks_first_and_keep_running_changes_nothing():
    at, analyzer, patcher, protocol_id, job_id = _start_running()
    try:
        at = _go_dashboard(at)
        at = _click(at, "home_add_btn")
        assert "nav_confirm_btn" in _keys(at) and "nav_cancel_btn" in _keys(at)
        assert "analysis is still running" in _texts(at)

        at = _click(at, "nav_cancel_btn")
        assert at.session_state["coord_view"] == "home"
        assert at.session_state["analysis_job_id"] == job_id
        assert at.session_state["coordinator_protocol_id"] == protocol_id
        assert analysis_jobs.jobs.get(job_id).active
        assert not at.session_state.get("_nav_pending")
        assert "home_resume_btn" in _keys(at)
    finally:
        _stop(analyzer, patcher)


def test_discard_and_continue_starts_clean_and_old_job_cannot_attach_to_new_protocol():
    at, analyzer, patcher, protocol_id, job_id = _start_running()
    try:
        at = _go_dashboard(at)
        at = _click(at, "home_add_btn")
        at = _click(at, "nav_confirm_btn")

        assert at.session_state["coord_view"] == "workflow"
        assert at.session_state["analysis_job_id"] is None
        assert at.session_state["coordinator_protocol_id"] is None
        assert at.session_state["uploaded_name"] is None
        assert at.session_state["pipeline_result"] is None
        assert analysis_jobs.jobs.get(job_id).status == analysis_jobs.DISCARDED
        assert at.file_uploader

        # Start a different protocol; then let the abandoned job finish.
        at = _upload(at, b"%PDF-1.4 other")
        new_id = at.session_state["coordinator_protocol_id"]
        assert new_id != protocol_id
        analyzer.release.set()
        analysis_jobs.jobs.wait(job_id)
        at = at.run()
        assert at.session_state["pipeline_result"] is None
        assert at.session_state["coordinator_protocol_id"] == new_id
        assert at.session_state["analysis_job_id"] is None
    finally:
        _stop(analyzer, patcher)


def test_discard_and_continue_to_saved_protocol_shows_only_that_protocol():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    at, analyzer, patcher, protocol_id, job_id = _start_running()
    try:
        at = _go_dashboard(at)
        at = _click(at, "home_view_btn")
        at = _click(at, "open_protocol_p-one")
        assert "nav_confirm_btn" in _keys(at)
        assert at.session_state["coordinator_protocol_id"] == protocol_id

        at = _click(at, "nav_confirm_btn")
        assert at.session_state["coordinator_protocol_id"] == "p-one"
        assert at.session_state["analysis_job_id"] is None
        assert analysis_jobs.jobs.get(job_id).status == analysis_jobs.DISCARDED
        assert all(
            i.plain_text.startswith("ONE:")
            for i in at.session_state["patient_guide"].all_items()
        )
        analyzer.release.set()
        analysis_jobs.jobs.wait(job_id)
        at = at.run()
        assert at.session_state["coordinator_protocol_id"] == "p-one"
        assert at.session_state["uploaded_name"] == "diabetes.pdf"
        assert at.session_state["pipeline_result"]["analysis"] is not None
        assert not at.exception
    finally:
        _stop(analyzer, patcher)


def test_keep_running_on_open_protocol_leaves_everything_in_place():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    at, analyzer, patcher, protocol_id, job_id = _start_running()
    try:
        at = _go_dashboard(at)
        at = _click(at, "home_view_btn")
        at = _click(at, "open_protocol_p-one")
        at = _click(at, "nav_cancel_btn")

        assert at.session_state["coordinator_protocol_id"] == protocol_id
        assert at.session_state["analysis_job_id"] == job_id
        assert analysis_jobs.jobs.get(job_id).active
        assert review_store.load_draft("p-one")["name"] == "Diabetes Study"
    finally:
        _stop(analyzer, patcher)


def test_analysis_that_finished_but_was_never_saved_asks_before_discard():
    analyzer = _GatedAnalyzer()
    analyzer.release.set()
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator", coord_view="home")
    at = _click(at.run(), "home_add_btn")
    at = _upload(at, b"%PDF-1.4 done")
    with patch("services.protocol_pipeline.analyze_protocol_pdf", analyzer):
        at = _click(at, "analyze_btn")
        analysis_jobs.jobs.wait(at.session_state["analysis_job_id"])
    at = at.run()
    assert at.session_state["pipeline_result"] is not None
    assert not review_store.list_protocols()

    at = _go_dashboard(at)
    assert "home_resume_btn" in _keys(at)
    at = _click(at, "home_add_btn")
    assert "nav_confirm_btn" in _keys(at)
    assert "not been saved as a protocol" in _texts(at)
    at = _click(at, "nav_cancel_btn")
    assert at.session_state["pipeline_result"] is not None
    at = _click(at, "home_add_btn")
    at = _click(at, "nav_confirm_btn")
    assert at.session_state["pipeline_result"] is None
    assert not review_store.list_protocols()


def test_uploaded_but_unanalyzed_pdf_asks_before_discard():
    at = _click(_dashboard(), "home_add_btn")
    at = _upload(at, b"%PDF-1.4 waiting")
    at = _go_dashboard(at)
    at = _click(at, "home_add_btn")

    assert "has not been analyzed" in _texts(at)
    at = _click(at, "nav_cancel_btn")
    assert at.session_state["uploaded_name"] == "protocol.pdf"


def test_back_to_dashboard_warns_about_typed_unsaved_edits():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    at = _click(_dashboard(), "home_view_btn")
    at = _click(at, "open_protocol_p-one")
    item = at.session_state["patient_guide"].all_items()[0]
    at.text_area(key=f"guide_{item.item_id}_text").set_value("A new wording I have not saved.")
    at = at.run()

    at = _go_dashboard(at)
    assert "nav_confirm_btn" in _keys(at)
    assert "statement edit" in _texts(at)
    assert at.session_state["coord_view"] == "workflow"
    at = _click(at, "nav_cancel_btn")
    assert at.session_state["coord_view"] == "workflow"
    assert at.text_area(key=f"guide_{item.item_id}_text").value == "A new wording I have not saved."

    at = _go_dashboard(at)
    at = _click(at, "nav_confirm_btn")
    assert at.session_state["coord_view"] == "home"
    stored = review_store.load_draft("p-one")["guide"].all_items()[0]
    assert stored.plain_text == item.plain_text


def test_unsaved_protocol_name_and_summary_text_are_flagged():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    review_store.save_original_summary("p-one", "Original summary.", "llm")
    at = _click(_dashboard(), "home_view_btn")
    at = _click(at, "open_protocol_p-one")
    at = at.run()
    at.session_state["summary_unsaved_p-one"] = "Edited but not saved."
    at = at.run()
    at = _go_dashboard(at)
    assert "summary edits you have not saved" in _texts(at)


def test_saved_and_unchanged_protocol_switches_without_a_prompt():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    _save("p-two", "Anemia Study", "anemia.pdf", marker="TWO")
    at = _click(_dashboard(), "home_view_btn")
    at = _click(at, "open_protocol_p-one")
    at = _go_dashboard(at)
    assert "nav_confirm_btn" not in _keys(at)
    assert "home_resume_btn" not in _keys(at)
    at = _click(at, "home_view_btn")
    at = _click(at, "open_protocol_p-two")
    assert "nav_confirm_btn" not in _keys(at)
    assert at.session_state["coordinator_protocol_id"] == "p-two"


def test_nothing_saved_is_deleted_by_discarding():
    _save("p-one", "Diabetes Study", "diabetes.pdf", marker="ONE")
    at, analyzer, patcher, protocol_id, job_id = _start_running()
    try:
        before = [e["protocol_id"] for e in review_store.list_protocols()]
        at = _go_dashboard(at)
        at = _click(at, "home_add_btn")
        at = _click(at, "nav_confirm_btn")
        assert [e["protocol_id"] for e in review_store.list_protocols()] == before
    finally:
        _stop(analyzer, patcher)


def test_nav_confirmation_is_unreachable_without_coordinator_account():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at.session_state["_nav_pending"] = {
        "kind": "add", "protocol_id": "", "typed": [], "other": ["x"],
    }
    at = at.run()
    assert "nav_confirm_btn" not in _keys(at)
    assert not at.file_uploader
