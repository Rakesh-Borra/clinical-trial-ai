"""Dashboard, upload-new-version flow, publication and patient visibility of versions."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from services import review_store, version_compare, version_store
from tests.auth_helpers import sign_in_as
from tests.publish_helpers import exact_rows, make_publishable
from tests.test_coordinator_dashboard import _dashboard, _keys
from tests.test_coordinator_dashboard import _texts as _base_texts
from tests.test_patient_guide_ui import (
    APP_PATH,
    _analyze,
    _click,
    _eight_statement_result,
    _fake_rewriter,
    _generated,
    _reviewed_session,
    _upload,
)


def _texts(at):
    return _base_texts(at) + " " + " ".join(
        [m.value for m in at.success] + [m.value for m in at.error]
    )


def _publish_v1():
    at, _, protocol_id = _reviewed_session()
    at = _click(at, "publish_protocol_btn")
    assert review_store.publication_status(protocol_id) == "published"
    return at, protocol_id


def _guide_pages(protocol_id):
    """Deterministic 'PDF text': each statement's excerpt on its cited page."""
    pages: dict[int, str] = {}
    for item in review_store.load_draft(protocol_id)["guide"].all_items():
        pages[item.source_page] = pages.get(item.source_page, "") + " " + item.original_text.casefold()
    return pages


def _open_version_picker(at, study_id):
    at = _click(at, "workflow_back_btn") if "workflow_back_btn" in _keys(at) else at
    at = _click(at, "home_version_btn")
    at.selectbox(key="version_study_select").select(study_id)
    return at.run()


def _build_v2(at, study_id, label="2.0", change=True):
    """Continue from the picker, upload an amended PDF, analyze, generate, approve."""
    at.text_input(key=f"version_label_{study_id}").set_value(label)
    at = at.run()
    at = _click(at, "version_continue_btn")
    assert at.session_state["coord_view"] == "workflow"
    at = _upload(at, b"%PDF-1.4 amended protocol")
    protocol_id = at.session_state["coordinator_protocol_id"]
    result = copy.deepcopy(_eight_statement_result())
    if change:
        result["analysis"].inclusion_criteria[0].description += " Amended wording."
        result["verification"] = exact_rows(result["analysis"])
    at = _click(at, "save_protocol_name_btn")
    at = _analyze(at, MagicMock(return_value=result))
    with patch("agents.plain_language_agent.ollama_rewriter", side_effect=_fake_rewriter):
        at = _click(at, "continue_review_btn")
        at = _generated(at)
    at = _click(at, "guide_approve_all_btn")
    for button in [b for b in at.button if b.key and b.key.endswith("_approve")]:
        at = _click(at, button.key)
    make_publishable(protocol_id)
    return at.run(), protocol_id


# --- dashboard ---------------------------------------------------------------

def test_dashboard_has_three_cards_in_order():
    at = _dashboard()
    keys = [b.key for b in at.button if b.key in {"home_add_btn", "home_version_btn", "home_view_btn"}]
    assert keys == ["home_add_btn", "home_version_btn", "home_view_btn"]
    text = _texts(at)
    assert text.index("Add New Protocol") < text.index("Upload New Version") < text.index("View Existing Protocols")


def test_upload_new_version_with_no_saved_trials_offers_add():
    at = _click(_dashboard(), "home_version_btn")
    assert "No clinical trials have been saved yet." in _texts(at)
    assert "version_add_btn" in _keys(at)


def test_picker_lists_trials_shows_current_version_and_blocks_duplicates():
    at, protocol_id = _publish_v1()
    study = version_store.get_version(protocol_id)["study_id"]
    at = _open_version_picker(at, study)
    text = _texts(at)
    assert "Currently published: **Version 1.0**" in text
    assert at.text_input(key=f"version_label_{study}").value == "2.0"  # suggested next
    for bad in ("1.0", "1.0 ", "bad label", ""):
        at.text_input(key=f"version_label_{study}").set_value(bad)
        at = at.run()
        assert next(b for b in at.button if b.key == "version_continue_btn").disabled, bad
    at.text_input(key=f"version_label_{study}").set_value("2.0")
    at = at.run()
    assert not next(b for b in at.button if b.key == "version_continue_btn").disabled


# --- end-to-end amendment ----------------------------------------------------------

def test_v2_upload_compare_approve_publish_and_patient_visibility():
    at, v1 = _publish_v1()
    study = version_store.get_version(v1)["study_id"]
    at = _open_version_picker(at, study)
    at, v2 = _build_v2(at, study)

    # V2 is a separate draft in the same study; V1 is untouched and still patient-visible.
    assert version_store.get_version(v2)["study_id"] == study
    assert version_store.get_version(v2)["version_label"] == "2.0"
    assert review_store.publication_status(v1) == "published"
    assert at.session_state["coordinator_protocols"][v2]["name"] == "Persistent Study"

    # Publication waits for the approved version comparison.
    publish_button = next(b for b in at.button if b.key == "publish_protocol_btn")
    assert publish_button.disabled
    assert "Compare this version with Version 1.0" in publish_button.help

    # Without stored PDFs the comparison is explicitly not source-verified.
    at = _click(at, "version_compare_btn")
    text = _texts(at)
    assert "Full source-backed version comparison is unavailable" in text
    assert "NOT been verified" in text
    assert "Not verified: original PDF unavailable" in text
    comparison = version_store.load_comparison(v2)
    assert comparison["source_verified"] == 0 and comparison["diff"]["changes"]

    # With both PDFs available the same changes are verified.
    with patch.object(version_compare, "default_page_texts", side_effect=lambda pid: _guide_pages(pid)):
        at = _click(at, "version_compare_btn")
    assert "Verified in both PDFs" in _texts(at)
    assert version_store.load_comparison(v2)["source_verified"] == 1
    assert next(b for b in at.button if b.key == "publish_protocol_btn").disabled  # not approved yet

    at = _click(at, "version_approve_btn")
    assert version_store.load_comparison(v2)["approved_at"]
    at = _click(at, "publish_protocol_btn")
    assert review_store.publication_status(v2) == "published"
    assert review_store.publication_status(v1) == "superseded"
    assert version_store.get_version(v2)["is_current"] == 1
    assert version_store.get_version(v1)["is_current"] == 0

    # Patients see only V2, labelled with its version and real publication date.
    patient = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(patient.session_state, "patient")
    patient = patient.run()
    options = patient.selectbox(key="patient_protocol_select").options
    assert options == ["Persistent Study"]
    patient.selectbox(key="patient_protocol_select").select("Persistent Study")
    patient = _click(patient.run(), "view_protocol_summary_btn")
    published_on = version_store.get_version(v2)["published_at"][:10]
    assert f"Version 2.0 · Published {published_on}" in _texts(patient)
    assert patient.session_state["patient_active_protocol_id"] == v2


def test_editing_statements_after_comparison_blocks_publication_again():
    at, v1 = _publish_v1()
    study = version_store.get_version(v1)["study_id"]
    at, v2 = _build_v2(_open_version_picker(at, study), study)
    with patch.object(version_compare, "default_page_texts", side_effect=lambda pid: _guide_pages(pid)):
        at = _click(at, "version_compare_btn")
    at = _click(at, "version_approve_btn")
    assert version_compare.comparison_block_reason(v2, at.session_state["patient_guide"]) is None
    guide = at.session_state["patient_guide"]
    guide.all_items()[0].plain_text += " Edited."
    assert "changed after the comparison" in version_compare.comparison_block_reason(v2, guide)


def test_draft_version_is_never_visible_to_patients():
    at, v1 = _publish_v1()
    study = version_store.get_version(v1)["study_id"]
    at, v2 = _build_v2(_open_version_picker(at, study), study, label="2.0")
    drafts_text = " ".join(i.plain_text for i in review_store.load_draft(v2)["guide"].all_items())
    patient = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(patient.session_state, "patient")
    patient = patient.run()
    patient.selectbox(key="patient_protocol_select").select("Persistent Study")
    patient = _click(patient.run(), "view_protocol_summary_btn")
    assert patient.session_state["patient_active_protocol_id"] == v1
    assert "Amended wording" not in _texts(patient) and "Amended wording" in drafts_text


def test_patient_chat_uses_only_the_current_published_version():
    at, v1 = _publish_v1()
    study = version_store.get_version(v1)["study_id"]
    at, v2 = _build_v2(_open_version_picker(at, study), study)
    with patch.object(version_compare, "default_page_texts", side_effect=lambda pid: _guide_pages(pid)):
        at = _click(at, "version_compare_btn")
    at = _click(at, "version_approve_btn")
    at = _click(at, "publish_protocol_btn")
    patient = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(patient.session_state, "patient")
    patient = patient.run()
    patient.selectbox(key="patient_protocol_select").select("Persistent Study")
    patient = _click(patient.run(), "view_protocol_summary_btn")
    records = patient.session_state["published_protocols"]
    assert set(records) == {v2}
    grounded = {i.plain_text for i in records[v2]["guide"].all_items()}
    old = {i.plain_text for i in review_store.load_publications()[v1]["guide"].all_items()}
    assert any("Amended wording" in t for t in grounded)
    assert not any("Amended wording" in t for t in old)


# --- list ---------------------------------------------------------------------------

def test_existing_protocols_are_grouped_by_trial_with_current_version_marked():
    at, v1 = _publish_v1()
    study = version_store.get_version(v1)["study_id"]
    at, v2 = _build_v2(_open_version_picker(at, study), study)
    at = _click(at, "workflow_back_btn")
    at = _click(at, "home_view_btn")
    text = _texts(at)
    assert text.count("Persistent Study") >= 1
    assert "Version 1.0" in text and "Version 2.0" in text
    assert "Current published version" in text
    assert f"open_protocol_{v1}" in _keys(at) and f"open_protocol_{v2}" in _keys(at)
    assert "Original PDF not stored" not in text  # both uploads were stored


def test_original_pdf_is_stored_with_hash_for_new_uploads():
    _, v1 = _publish_v1()
    version = version_store.get_version(v1)
    assert version["pdf_status"] == "stored" and version["pdf_sha256"]
    assert version_store.stored_pdf_bytes(v1) == b"%PDF-1.4 protocol A"


def test_legacy_protocol_without_pdf_can_reattach_only_the_original():
    review_store.save_draft(
        "legacy", __import__("tests.version_helpers", fromlist=["x"]).make_guide(),
        name="Legacy Study", uploaded_name="old.pdf",
        upload_fingerprint="old.pdf:12:" + __import__("hashlib").sha256(b"%PDF-legacy!").hexdigest()[:16],
    )
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "coordinator", coord_view="list")
    at = at.run()
    assert "Original PDF not stored for this version." in _texts(at)
    at.file_uploader(key="reattach_upload_legacy").set_value(("x.pdf", b"%PDF-wrong!!", "application/pdf"))
    at = _click(at.run(), "reattach_btn_legacy")
    assert version_store.get_version("legacy")["pdf_status"] == "unavailable"
    at.file_uploader(key="reattach_upload_legacy").set_value(("old.pdf", b"%PDF-legacy!", "application/pdf"))
    at = _click(at.run(), "reattach_btn_legacy")
    assert version_store.get_version("legacy")["pdf_status"] == "stored"
    assert len(version_store.list_studies()) == 1  # no new version was created


# --- authorization ------------------------------------------------------------------

def test_patient_cannot_reach_version_pages_or_store_functions():
    from services import authz

    _, v1 = _publish_v1()
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(at.session_state, "patient")
    at.session_state["coord_view"] = "versions"
    at = at.run()
    assert not {"version_continue_btn", "home_version_btn"} & _keys(at)
    assert not at.file_uploader

    # Direct calls inside a patient session are refused too.
    import streamlit.runtime.scriptrunner as sr

    with patch.object(sr, "get_script_run_ctx", return_value=object()), patch.object(
        authz, "require_coordinator", side_effect=authz.AuthorizationError("no")
    ):
        for call in (
            lambda: version_store.list_studies(),
            lambda: version_store.reattach_pdf(v1, b"x", "x.pdf"),
            lambda: version_store.approve_comparison(v1),
            lambda: version_compare.compare_versions(v1),
        ):
            with pytest.raises(authz.AuthorizationError):
                call()
