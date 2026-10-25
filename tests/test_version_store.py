"""Protocol versions: migration, uniqueness, atomic publication, PDFs, authorization."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing

import pytest

from models.plain_language_models import PatientGuide
from services import review_store, version_store
from services.version_store import VersionError
from tests.version_helpers import make_guide

LEGACY_SCHEMA = review_store._SCHEMA


def _guide() -> PatientGuide:
    return make_guide()


def _fingerprint(data: bytes, name: str = "p.pdf") -> str:
    return f"{name}:{len(data)}:{hashlib.sha256(data).hexdigest()[:16]}"


def _save(pid, name="Diabetes Study", *, study_id=None, label=None, data=b"%PDF-1 v1"):
    review_store.save_draft(
        pid, _guide(), name=name, uploaded_name="p.pdf",
        upload_fingerprint=_fingerprint(data), study_id=study_id, version_label=label,
    )


def _publish(pid, name="Diabetes Study"):
    review_store.publish(pid, name, _guide())


def _study_of(pid):
    return version_store.get_version(pid)["study_id"]


# --- migration -----------------------------------------------------------

def _legacy_db(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(LEGACY_SCHEMA)
        g = _guide().model_dump_json()
        conn.execute("INSERT INTO drafts VALUES ('old-pub','Old Pub','a.pdf','fp1',NULL,?, '2026-01-01T00:00:00+00:00')", (g,))
        conn.execute("INSERT INTO drafts VALUES ('old-draft','Old Draft','b.pdf','fp2',NULL,?, '2026-01-02T00:00:00+00:00')", (g,))
        conn.execute("INSERT INTO publications VALUES ('old-pub','Old Pub',?,'published','2026-01-03T00:00:00+00:00',NULL)", (g,))
        conn.execute("INSERT INTO history (protocol_id, at, action, detail) VALUES ('old-pub','t','published','')")


def test_migration_backs_up_and_preserves_existing_records(tmp_path):
    _legacy_db(tmp_path / "ctai_test.db")
    review_store.load_publications()  # first use triggers the migration
    backups = list((tmp_path / "backups").glob("ctai-pre-versioning-*.db"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as conn:
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 2
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name='protocol_versions'"
        ).fetchone() is None  # backup is the pre-migration state

    with closing(sqlite3.connect(tmp_path / "ctai_test.db")) as conn:
        assert conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM publications").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 1
    pub = version_store.get_version("old-pub")
    assert (pub["version_label"], pub["is_current"], pub["pdf_status"]) == (
        "1.0", 1, version_store.PDF_UNAVAILABLE)
    assert pub["pdf_path"] is None and pub["pdf_sha256"] is None  # nothing fabricated
    assert version_store.get_version("old-draft")["is_current"] == 0
    assert version_store.get_version("old-pub")["study_id"] != version_store.get_version("old-draft")["study_id"]


def test_migration_runs_once_and_empty_database_needs_no_backup(tmp_path):
    review_store.load_publications()
    review_store.load_publications()
    assert not (tmp_path / "backups").exists()
    _legacy_db(tmp_path / "second.db")
    import os
    os.environ["CTAI_DB_PATH"] = str(tmp_path / "second.db")
    review_store.load_publications()
    review_store.load_publications()
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1


# --- versions ------------------------------------------------------------

def test_first_save_creates_study_with_default_version():
    _save("p1")
    v = version_store.get_version("p1")
    assert v["version_label"] == "1.0" and v["study_title"] == "Diabetes Study"


def test_explicit_label_and_validation():
    _save("p1", label="A-2.1")
    assert version_store.get_version("p1")["version_label"] == "A-2.1"
    for bad in ("", "has space", "x" * 21, "v1/2", "é1"):
        with pytest.raises(VersionError):
            version_store.validate_label(bad)
    assert version_store.validate_label("x" * 20) == "x" * 20


def test_second_version_joins_study_and_label_is_unique_per_study():
    _save("p1")
    study = _study_of("p1")
    _save("p2", study_id=study, label="2.0")
    assert version_store.sibling_ids("p1") == {"p2"}
    with pytest.raises(VersionError):
        _save("p3", study_id=study, label="2.0")
    with pytest.raises(VersionError):
        _save("p3", study_id=study, label="2.0".upper().lower())
    assert not review_store.draft_exists("p3")  # nothing half-saved
    _save("other", name="Other Study")  # the same label is fine in another study
    assert version_store.get_version("other")["version_label"] == "1.0"


def test_unknown_study_rejected():
    with pytest.raises(VersionError):
        _save("p1", study_id="study-missing", label="2.0")
    assert not review_store.draft_exists("p1")


def test_resaving_a_version_keeps_its_identity():
    _save("p1", label="3.0")
    _save("p1", label="9.9")
    assert version_store.get_version("p1")["version_label"] == "3.0"


# --- publication ---------------------------------------------------------

def test_publishing_v2_supersedes_v1_and_keeps_history():
    _save("v1")
    study = _study_of("v1")
    _publish("v1")
    _save("v2", study_id=study, label="2.0")
    pubs = review_store.load_publications()
    assert pubs["v1"]["status"] == "published"  # V1 stays visible while V2 is a draft
    assert "v2" not in pubs

    _publish("v2")
    pubs = review_store.load_publications()
    assert pubs["v2"]["status"] == "published"
    assert pubs["v1"]["status"] == "superseded"  # snapshot retained as history
    assert pubs["v1"]["guide"] is not None
    assert version_store.get_version("v1")["is_current"] == 0
    assert version_store.get_version("v1")["superseded_at"]
    assert pubs["v2"]["version_label"] == "2.0" and pubs["v2"]["published_at"]


def test_v3_and_only_one_current_version_per_study():
    _save("v1")
    study = _study_of("v1")
    for pid, label in (("v2", "2.0"), ("v3", "3.0")):
        _save(pid, study_id=study, label=label)
    for pid in ("v1", "v2", "v3"):
        _publish(pid)
    current = [v for v in version_store.list_studies()[0]["versions"] if v["is_current"]]
    assert [v["protocol_id"] for v in current] == ["v3"]
    statuses = {k: v["status"] for k, v in review_store.load_publications().items()}
    assert statuses == {"v1": "superseded", "v2": "superseded", "v3": "published"}


def test_database_rejects_two_current_versions_in_one_study():
    _save("v1")
    _save("v2", study_id=_study_of("v1"), label="2.0")
    with closing(sqlite3.connect(review_store.db_path())) as conn:
        conn.execute("UPDATE protocol_versions SET is_current=1 WHERE protocol_id='v1'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE protocol_versions SET is_current=1 WHERE protocol_id='v2'")


def test_studies_publish_independently():
    _save("a", name="A")
    _save("b", name="B")
    _publish("a", "A")
    _publish("b", "B")
    assert {k: v["status"] for k, v in review_store.load_publications().items()} == {
        "a": "published", "b": "published"}


def test_unpublish_current_leaves_no_current_and_keeps_superseded():
    _save("v1")
    _save("v2", study_id=_study_of("v1"), label="2.0")
    _publish("v1")
    _publish("v2")
    review_store.unpublish("v2")
    assert review_store.load_publications()["v2"]["status"] == "withdrawn"
    assert review_store.load_publications()["v1"]["status"] == "superseded"
    assert version_store.list_studies()[0]["current"] is None


def test_failed_publish_changes_nothing(monkeypatch):
    _save("v1")
    _save("v2", study_id=_study_of("v1"), label="2.0")
    _publish("v1")

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(version_store, "promote", boom)
    with pytest.raises(sqlite3.OperationalError):
        _publish("v2")
    pubs = review_store.load_publications()
    assert "v2" not in pubs and pubs["v1"]["status"] == "published"
    assert version_store.get_version("v1")["is_current"] == 1


def test_summary_and_approval_records_stay_per_version():
    _save("v1")
    _save("v2", study_id=_study_of("v1"), label="2.0")
    review_store.save_original_summary("v1", "Version one summary", "llm")
    review_store.approve_summary("v1")
    review_store.save_original_summary("v2", "Version two summary", "llm")
    _publish("v1")
    _publish("v2")
    assert review_store.load_summary("v1")["approved_text"] == "Version one summary"
    assert review_store.load_summary("v2")["approved_text"] is None
    assert review_store.load_summary("v2")["draft_text"] == "Version two summary"


# --- listing --------------------------------------------------------------

def test_list_studies_groups_versions_and_marks_current():
    _save("v1")
    _save("v2", study_id=_study_of("v1"), label="2.0")
    _save("x", name="Other")
    _publish("v1")
    studies = {s["title"]: s for s in version_store.list_studies()}
    assert {v["version_label"] for v in studies["Diabetes Study"]["versions"]} == {"1.0", "2.0"}
    assert studies["Diabetes Study"]["current"]["protocol_id"] == "v1"
    assert studies["Other"]["current"] is None


# --- original PDFs ----------------------------------------------------------

def test_attach_pdf_stores_once_with_hash_outside_git(tmp_path):
    data = b"%PDF-1 original"
    _save("p1", data=data)
    assert version_store.attach_pdf("p1", data, "p.pdf") is True
    v = version_store.get_version("p1")
    assert v["pdf_status"] == "stored" and v["pdf_sha256"] == hashlib.sha256(data).hexdigest()
    assert str(tmp_path) in v["pdf_path"] and not list(tmp_path.glob("**/*.part"))
    assert version_store.stored_pdf_bytes("p1") == data
    assert version_store.attach_pdf("p1", b"%PDF-1 different", "x.pdf") is False  # never overwritten
    assert version_store.stored_pdf_bytes("p1") == data


def test_tampered_stored_pdf_is_not_trusted():
    data = b"%PDF-1 original"
    _save("p1", data=data)
    version_store.attach_pdf("p1", data, "p.pdf")
    open(version_store.get_version("p1")["pdf_path"], "wb").write(b"tampered")
    assert version_store.stored_pdf_bytes("p1") is None


def test_reattach_validates_fingerprint_and_creates_no_version():
    data = b"%PDF-1 the real original"
    _save("p1", data=data)
    before = version_store.list_studies()
    with pytest.raises(VersionError):
        version_store.reattach_pdf("p1", b"%PDF-1 some other file", "o.pdf")
    assert version_store.get_version("p1")["pdf_status"] == "unavailable"
    version_store.reattach_pdf("p1", data, "renamed.pdf")  # a renamed copy is still the original
    assert version_store.get_version("p1")["pdf_status"] == "stored"
    assert version_store.list_studies() == [
        {**before[0], "versions": [{**before[0]["versions"][0], **{
            k: version_store.get_version("p1")[k] for k in ("pdf_status", "pdf_path", "pdf_sha256", "pdf_name")}}]}
    ]
    with pytest.raises(VersionError):
        version_store.reattach_pdf("p1", data, "again.pdf")


def test_reattach_without_saved_fingerprint_is_refused(tmp_path):
    _legacy_db(tmp_path / "ctai_test.db")
    with closing(sqlite3.connect(tmp_path / "ctai_test.db")) as conn, conn:
        conn.execute("UPDATE drafts SET upload_fingerprint = NULL WHERE protocol_id='old-draft'")
    with pytest.raises(VersionError):
        version_store.reattach_pdf("old-draft", b"%PDF-1", "b.pdf")
    assert version_store.get_version("old-draft")["pdf_status"] == "unavailable"


def test_failed_pdf_write_leaves_no_partial_file(tmp_path, monkeypatch):
    _save("p1", data=b"%PDF-1 x")

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", fail)
    with pytest.raises(OSError):
        version_store.attach_pdf("p1", b"%PDF-1 x", "p.pdf")
    assert not list(tmp_path.glob("**/*.part")) and not list(tmp_path.glob("**/p1.pdf"))
    assert version_store.get_version("p1")["pdf_status"] == "unavailable"
