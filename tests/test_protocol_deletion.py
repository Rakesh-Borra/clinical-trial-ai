"""Deleting protocol versions: confirmation, authorization, isolation, atomicity."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from services import auth, protocol_deletion, review_store, version_store
from services.protocol_deletion import DeletionError
from tests.auth_helpers import sign_in_as
from tests.test_patient_guide_ui import APP_PATH
from tests.version_helpers import make_guide

PASSWORD = "Passw0rd-test-123"
NAME = "Diabetes  Study"  # double space: spacing must match exactly
COORD = "coord@example.test"


@pytest.fixture(autouse=True)
def coordinator():
    return auth.create_user("Coord", COORD, PASSWORD, "coordinator")


def _save(pid, *, study_id=None, label="1.0", pdf=True):
    data = f"%PDF-1 {pid}".encode()
    review_store.save_draft(
        pid, make_guide(), name=NAME, uploaded_name="p.pdf",
        upload_fingerprint=f"p.pdf:{len(data)}:x", study_id=study_id, version_label=label,
    )
    if pdf:
        version_store.attach_pdf(pid, data, "p.pdf")
    return version_store.get_version(pid)["study_id"]


def _rows(table, pid):
    with closing(review_store._connect()) as conn:
        return conn.execute(f"SELECT * FROM {table} WHERE protocol_id = ?", (pid,)).fetchall()


def _delete(pid, **kw):
    args = dict(typed_name=NAME, password=PASSWORD, actor_email=COORD)
    args.update(kw)
    return protocol_deletion.delete_protocol(pid, **args)


def _two_versions():
    study = _save("v1")
    _save("v2", study_id=study, label="2.0")
    return study


def test_deletes_only_the_selected_version_and_its_pdf():
    _two_versions()
    pdf = Path(version_store.get_version("v1")["pdf_path"])
    result = _delete("v1")
    assert result["deleted_protocol_ids"] == ["v1"]
    assert not pdf.exists() and not list(pdf.parent.glob("*.deleting"))
    for table in ("drafts", "protocol_versions"):
        assert _rows(table, "v1") == []
    assert version_store.get_version("v2") is not None
    assert review_store.load_draft("v2") is not None
    assert Path(version_store.get_version("v2")["pdf_path"]).exists()


def test_other_studies_are_untouched():
    _save("a")
    other = _save("b", label="1.0")
    review_store.save_draft("b", make_guide(), name="Other", version_label="1.0")
    _delete("a")
    assert version_store.get_version("b")["study_id"] == other
    assert review_store.load_draft("b") is not None


def test_whole_study_deletion_removes_every_version():
    study = _two_versions()
    _delete("v1", scope=protocol_deletion.SCOPE_STUDY)
    assert version_store.get_version("v1") is None and version_store.get_version("v2") is None
    assert version_store.study_title(study) is None


def test_deleting_the_last_version_removes_the_empty_study():
    study = _save("only")
    _delete("only")
    assert version_store.study_title(study) is None


def test_wrong_name_is_refused_and_nothing_changes():
    _save("v1")
    for typed in ("diabetes  study", "Diabetes Study", " Diabetes  Study", ""):
        with pytest.raises(DeletionError):
            _delete("v1", typed_name=typed)
    assert version_store.get_version("v1") is not None


def test_wrong_password_is_refused_and_nothing_changes():
    _save("v1")
    with pytest.raises(DeletionError, match="password"):
        _delete("v1", password="wrong-password")
    with pytest.raises(DeletionError):
        _delete("v1", password="")
    assert Path(version_store.get_version("v1")["pdf_path"]).exists()
    assert review_store.load_draft("v1") is not None


def test_repeated_wrong_passwords_lock_out_deletion():
    _save("v1")
    for _ in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(DeletionError):
            _delete("v1", password="wrong-password")
    with pytest.raises(DeletionError, match="Too many"):
        _delete("v1")
    assert version_store.get_version("v1") is not None


def test_requires_a_coordinator_account():
    _save("v1")
    auth.create_user("Pat", "pat@example.test", PASSWORD, "patient")
    for actor in (None, "pat@example.test", "nobody@example.test"):
        with pytest.raises(PermissionError):
            _delete("v1", actor_email=actor)
    assert version_store.get_version("v1") is not None


def test_published_version_deletion_removes_patient_access_without_republishing():
    study = _two_versions()
    review_store.publish("v1", NAME, make_guide())
    review_store.publish("v2", NAME, make_guide())  # v2 current, v1 superseded
    result = _delete("v2")
    assert result["was_published"] is True
    assert review_store.publication_status("v2") is None
    assert review_store.publication_status("v1") == review_store.SUPERSEDED
    assert not any(v["is_current"] for v in version_store.list_studies()[0]["versions"])
    assert version_store.get_version("v1")["study_id"] == study


def test_audit_trail_records_deletion_without_password():
    _save("v1")
    _delete("v1")
    with closing(review_store._connect()) as conn:
        audit = [dict(r) for r in conn.execute("SELECT * FROM deletion_audit")]
        dump = " ".join(str(v) for r in audit for v in r.values())
        history = [r["action"] for r in conn.execute("SELECT action FROM history WHERE protocol_id='v1'")]
    assert len(audit) == 1 and audit[0]["user_email"] == COORD
    assert audit[0]["scope"] == "version" and "v1" in audit[0]["protocol_ids_json"]
    assert PASSWORD not in dump
    assert "protocol_deleted" in history


def test_database_failure_rolls_back_and_restores_the_pdf(monkeypatch):
    _two_versions()
    pdf = Path(version_store.get_version("v1")["pdf_path"])
    real = sqlite3.Connection

    class Failing(real):
        def execute(self, sql, *a):
            if sql.startswith("INSERT INTO deletion_audit"):
                raise sqlite3.OperationalError("disk I/O error")
            return super().execute(sql, *a)

    original = review_store._connect
    monkeypatch.setattr(review_store, "_connect", lambda: _connect_failing(Failing))
    with pytest.raises(DeletionError, match="Nothing was deleted"):
        _delete("v1")
    monkeypatch.setattr(review_store, "_connect", original)
    assert pdf.exists() and not list(pdf.parent.glob("*.deleting"))
    assert version_store.get_version("v1") is not None and review_store.load_draft("v1")
    assert _rows("drafts", "v1") != []


def _connect_failing(factory):
    path = review_store.db_path()
    conn = sqlite3.connect(path, timeout=10, factory=factory)
    conn.row_factory = sqlite3.Row
    return conn


def test_pdf_outside_the_pdf_directory_is_never_deleted(tmp_path):
    _save("v1", pdf=False)
    outside = tmp_path / "elsewhere.pdf"
    outside.write_bytes(b"keep me")
    with closing(review_store._connect()) as conn, conn:
        conn.execute("UPDATE protocol_versions SET pdf_path = ? WHERE protocol_id='v1'", (str(outside),))
    _delete("v1")
    assert outside.read_bytes() == b"keep me"


# --- dashboard ------------------------------------------------------------

def _list_view(role="coordinator"):
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.session_state["role"] = role
    sign_in_as(at.session_state, role, email=COORD if role == "coordinator" else None, coord_view="list")
    return at.run()


def _dialog_app(pid):
    """The dialog body on its own page (AppTest cannot keep a modal open across reruns)."""

    def script(pid):
        import streamlit as st

        from services import review_store
        from ui import coordinator_home

        entry = next(e for e in review_store.list_protocols() if e["protocol_id"] == pid)
        coordinator_home._confirm_delete.__wrapped__(entry)
        if st.button("again"):
            st.rerun()

    at = AppTest.from_function(script, args=(pid,), default_timeout=60)
    sign_in_as(at.session_state, "coordinator", email=COORD)
    return at.run()


def test_list_offers_a_delete_action_for_each_protocol():
    _two_versions()
    keys = {b.key for b in _list_view().button}
    assert {"delete_protocol_v1", "delete_protocol_v2"} <= keys


def test_dialog_shows_exact_name_version_and_defaults_to_one_version():
    _two_versions()
    at = _dialog_app("v1")
    assert at.radio(key="delete_scope_v1").value == "version"
    text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning)
    assert "`Diabetes  Study`" in text and "**1.0**" in text
    assert "version 1.0" in " ".join(w.value for w in at.warning)


def test_dialog_enables_delete_only_with_name_and_password():
    _save("v1")
    at = _dialog_app("v1")
    assert at.button(key="delete_confirm_v1").disabled
    at.text_input(key="delete_name_v1").set_value("wrong")
    at.text_input(key="delete_password_v1").set_value(PASSWORD)
    at = at.run()
    assert at.button(key="delete_confirm_v1").disabled
    at.text_input(key="delete_name_v1").set_value(NAME)
    at = at.run()
    assert not at.button(key="delete_confirm_v1").disabled
    at.button(key="delete_confirm_v1").click()
    at.run()
    assert version_store.get_version("v1") is None


def test_dialog_wrong_password_shows_error_and_keeps_protocol():
    _save("v1")
    at = _dialog_app("v1")
    at.text_input(key="delete_name_v1").set_value(NAME)
    at.text_input(key="delete_password_v1").set_value("not-my-password")
    at = at.run()
    at.button(key="delete_confirm_v1").click()
    at = at.run()
    assert any("password" in e.value for e in at.error)
    assert version_store.get_version("v1") is not None


def test_dialog_warns_that_deleting_the_published_version_removes_patient_access():
    _save("v1")
    review_store.publish("v1", NAME, make_guide())
    at = _dialog_app("v1")
    assert any("patients will lose access" in e.value for e in at.error)


def test_patient_cannot_reach_delete_controls():
    _save("v1")
    at = _list_view(role="patient")
    assert not [b for b in at.button if (b.key or "").startswith("delete_")]
    assert version_store.get_version("v1") is not None


# --- deleting V1 while V2 is current and published ------------------------

V2_CONTENT = {"visits": [("Amended wording: visits every 3 weeks.", 4, "Visits every 3 weeks.")]}


def _v1_then_published_v2():
    study = _save("v1")
    review_store.publish("v1", NAME, make_guide())
    _save("v2", study_id=study, label="2.0")
    v2_guide = make_guide(V2_CONTENT)
    review_store.save_draft("v2", v2_guide, name=NAME, uploaded_name="p.pdf",
                            upload_fingerprint="p.pdf:9:x", study_id=study, version_label="2.0")
    review_store.publish("v2", NAME, v2_guide)
    return study


def _snapshot(pid):
    with closing(review_store._connect()) as conn:
        out = {}
        for table in ("drafts", "publications", "summaries", "protocol_versions", "version_comparisons"):
            out[table] = [tuple(r) for r in conn.execute(f"SELECT * FROM {table} WHERE protocol_id = ?", (pid,))]
        return out


def test_deleting_v1_leaves_published_v2_unchanged():
    study = _v1_then_published_v2()
    before = _snapshot("v2")
    pdf2 = Path(version_store.get_version("v2")["pdf_path"])
    pdf2_bytes = pdf2.read_bytes()
    _delete("v1")
    assert _snapshot("v2") == before
    assert pdf2.read_bytes() == pdf2_bytes
    v2 = version_store.get_version("v2")
    assert v2["is_current"] == 1 and v2["version_label"] == "2.0"
    assert review_store.publication_status("v2") == review_store.PUBLISHED
    studies = version_store.list_studies()
    assert [v["protocol_id"] for v in studies[0]["versions"]] == ["v2"]
    assert studies[0]["current"]["protocol_id"] == "v2" and studies[0]["study_id"] == study


def test_patients_and_qa_use_only_v2_after_v1_is_deleted():
    _v1_then_published_v2()
    _delete("v1")
    published = {pid: rec for pid, rec in review_store.load_publications().items()
                 if rec["status"] == review_store.PUBLISHED}
    assert set(published) == {"v2"}
    texts = {i.plain_text for i in published["v2"]["guide"].all_items()}
    assert any("Amended wording" in t for t in texts)
    patient = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(patient.session_state, "patient")
    patient = patient.run()
    assert set(patient.session_state["published_protocols"]) == {"v2"}
    assert not [e for e in patient.exception]


def test_v2_still_publishes_and_opens_after_its_comparison_partner_is_deleted():
    from services import version_compare

    _v1_then_published_v2()
    with closing(review_store._connect()) as conn, conn:
        conn.execute(
            "INSERT INTO version_comparisons (protocol_id, previous_protocol_id, diff_json, "
            "source_verified, created_at) VALUES ('v2', 'v1', '{}', 0, 't')"
        )
    _delete("v1")
    assert version_store.previous_version("v2") is None
    assert version_compare.comparison_block_reason("v2", make_guide(V2_CONTENT)) is None
    review_store.publish("v2", NAME, make_guide(V2_CONTENT))  # republish is not blocked
    assert review_store.publication_status("v2") == review_store.PUBLISHED


def test_deleting_published_v2_leaves_published_v1_untouched():
    study = _save("v1")
    review_store.publish("v1", NAME, make_guide())
    _save("v2", study_id=study, label="2.0")
    before = _snapshot("v1")
    pdf1 = Path(version_store.get_version("v1")["pdf_path"])
    _delete("v2")
    assert _snapshot("v1") == before and pdf1.exists()
    assert version_store.get_version("v1")["is_current"] == 1


def test_deleting_current_v2_does_not_republish_superseded_v1():
    _v1_then_published_v2()
    assert review_store.publication_status("v1") == review_store.SUPERSEDED
    _delete("v2")
    assert review_store.publication_status("v1") == review_store.SUPERSEDED
    assert version_store.get_version("v1")["is_current"] == 0
    assert {p for p, r in review_store.load_publications().items()
            if r["status"] == review_store.PUBLISHED} == set()


# --- published name reuse after deletion ------------------------------------


def test_clear_session_after_delete_drops_published_name_cache():
    guide = make_guide()
    session = {
        "published_protocols": {
            "gone": {"name": NAME, "guide": guide},
            "keep": {"name": "Other Study", "guide": guide},
        },
        "coordinator_protocols": {
            "gone": {"name": NAME},
            "keep": {"name": "Other Study"},
        },
        "patient_active_protocol_id": "gone",
        "patient_history": [{"query": "hi"}],
    }
    protocol_deletion.clear_session_after_delete(session, ["gone"])
    assert "gone" not in session["published_protocols"]
    assert "keep" in session["published_protocols"]
    assert "gone" not in session["coordinator_protocols"]
    assert session["patient_active_protocol_id"] is None
    assert session["patient_history"] == []


def test_deleted_published_name_is_reusable_and_self_publish_is_not_a_conflict(
    monkeypatch,
):
    import app

    _save("old")
    review_store.publish("old", NAME, make_guide())
    _save("new")

    class _State(dict):
        pass

    state = _State()
    state.update(
        {
            "coordinator_protocol_id": "new",
            "coord_study_id": None,
            "published_protocols": {
                "old": {"name": NAME, "guide": make_guide()},
            },
            "coordinator_protocols": {"new": {"name": NAME}, "old": {"name": NAME}},
            "patient_active_protocol_id": None,
        }
    )
    monkeypatch.setattr(app.st, "session_state", state)

    # Stale session still lists the old publication, but DB says it is active —
    # name must stay blocked for a different study.
    assert app._other_active_published_uses_name("new", NAME) is True

    result = _delete("old")
    protocol_deletion.clear_session_after_delete(
        state, result["deleted_protocol_ids"]
    )
    app.sync_published_from_store()

    assert "old" not in state["published_protocols"]
    assert review_store.publication_status("old") is None
    # Name is free for the remaining protocol.
    assert app._other_active_published_uses_name("new", NAME) is False
    # Re-publishing / checking the same id never conflicts with itself.
    review_store.publish("new", NAME, make_guide())
    state["coordinator_protocol_id"] = "new"
    state["published_protocols"]["new"] = {"name": NAME, "guide": make_guide()}
    assert app._other_active_published_uses_name("new", NAME) is False


def test_deleted_unpublished_name_is_reusable(monkeypatch):
    import app

    _save("draft-only")
    _save("other")
    state = {
        "coordinator_protocol_id": "other",
        "coord_study_id": None,
        "published_protocols": {},
        "coordinator_protocols": {
            "draft-only": {"name": NAME},
            "other": {"name": NAME},
        },
        "patient_active_protocol_id": None,
    }
    monkeypatch.setattr(app.st, "session_state", state)
    assert app._other_active_published_uses_name("other", NAME) is False
    _delete("draft-only")
    protocol_deletion.clear_session_after_delete(state, ["draft-only"])
    assert "draft-only" not in state["coordinator_protocols"]
    assert app._other_active_published_uses_name("other", NAME) is False


def test_active_published_duplicate_name_across_studies_remains_blocked(monkeypatch):
    import app

    _save("study-a")
    review_store.publish("study-a", NAME, make_guide())
    _save("study-b")
    state = {
        "coordinator_protocol_id": "study-b",
        "coord_study_id": None,
        "published_protocols": {
            "study-a": {"name": NAME, "guide": make_guide()},
        },
        "coordinator_protocols": {
            "study-a": {"name": NAME},
            "study-b": {"name": NAME},
        },
        "patient_active_protocol_id": None,
    }
    monkeypatch.setattr(app.st, "session_state", state)
    assert app._other_active_published_uses_name("study-b", NAME) is True


def test_stale_session_entry_does_not_block_name_after_db_delete(monkeypatch):
    """Even before session clear, publish name checks trust the database."""
    import app

    _save("old")
    review_store.publish("old", NAME, make_guide())
    _save("new")
    state = {
        "coordinator_protocol_id": "new",
        "coord_study_id": None,
        "published_protocols": {
            "old": {"name": NAME, "guide": make_guide()},
        },
        "coordinator_protocols": {"new": {"name": NAME}},
        "patient_active_protocol_id": None,
    }
    monkeypatch.setattr(app.st, "session_state", state)
    assert app._other_active_published_uses_name("new", NAME) is True
    _delete("old")
    # Session still has the stale "old" entry, but the DB row is gone.
    assert "old" in state["published_protocols"]
    assert review_store.publication_status("old") is None
    assert app._other_active_published_uses_name("new", NAME) is False


def test_restart_loads_only_remaining_published_after_delete():
    _save("old")
    review_store.publish("old", NAME, make_guide())
    _save("keeper")
    review_store.publish("keeper", "Keeper Study", make_guide())
    _delete("old")
    patient = AppTest.from_file(APP_PATH, default_timeout=30)
    sign_in_as(patient.session_state, "patient")
    patient = patient.run()
    published = patient.session_state["published_protocols"]
    assert "old" not in published
    assert "keeper" in published
    assert published["keeper"]["name"] == "Keeper Study"
