"""
Protocol version management on top of the existing review tables.

Every saved protocol (a ``protocol_id`` in drafts, publications, summaries,
...) is one *version* of a *study*. Because each version keeps its own
protocol_id, its analysis, statements, summary, approval and publication
snapshot stay separate and are never overwritten by a later amendment.

Added tables (additive, created on first use):

- studies:             stable study id and title
- protocol_versions:   one row per protocol_id with the study it belongs to, a
                       version label (unique within the study), the original PDF
                       reference and hash, and the current-published flag. A
                       partial unique index allows only one current version per
                       study.
- version_comparisons: the saved comparison of a version with its predecessor,
                       and the coordinator's approval of it.

Existing databases are migrated in place: a backup is written first, every
saved protocol becomes version "1.0" of its own study, and no row is altered
or removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.review_store import _coordinator_only

SUPERSEDED = "superseded"
DEFAULT_LABEL = "1.0"
SCHEMA_VERSION = 1
PDF_STORED = "stored"
PDF_UNAVAILABLE = "unavailable"

_LABEL_RE = re.compile(r"^[A-Za-z0-9.\-]{1,20}$")
LABEL_RULES = "Use 1–20 letters, numbers, dots or hyphens (for example 1.0 or 2.1-rc1)."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    study_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS protocol_versions (
    protocol_id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES studies(study_id),
    version_label TEXT NOT NULL COLLATE NOCASE,
    pdf_status TEXT NOT NULL DEFAULT 'unavailable',
    pdf_name TEXT,
    pdf_path TEXT,
    pdf_sha256 TEXT,
    uploaded_at TEXT,
    published_at TEXT,
    superseded_at TEXT,
    is_current INTEGER NOT NULL DEFAULT 0,
    UNIQUE (study_id, version_label)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_current_version_per_study
    ON protocol_versions (study_id) WHERE is_current = 1;
CREATE TABLE IF NOT EXISTS version_comparisons (
    protocol_id TEXT PRIMARY KEY,
    previous_protocol_id TEXT NOT NULL,
    diff_json TEXT NOT NULL,
    source_verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    approved_at TEXT,
    approved_by TEXT
);
"""


class VersionError(ValueError):
    """A user-facing problem with a version identifier, upload or comparison."""


def _now() -> str:
    from services import review_store

    return review_store._now()


def _connect() -> sqlite3.Connection:
    from services import review_store

    return review_store._connect()


# ---------------------------------------------------------------------------
# Schema, backup and migration
# ---------------------------------------------------------------------------

def backup_dir() -> Path:
    from services import review_store

    return Path(os.getenv("CTAI_BACKUP_DIR") or review_store.db_path().parent / "backups")


def backup_database(conn: sqlite3.Connection, reason: str) -> Path:
    """Consistent copy of the database (SQLite online backup); never overwrites."""
    target_dir = backup_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now().replace(":", "").replace("-", "").replace("+0000", "")
    target = target_dir / f"ctai-{reason}-{stamp}-{uuid.uuid4().hex[:6]}.db"
    with closing(sqlite3.connect(target)) as dest:
        conn.backup(dest)
    return target


def _has_saved_records(conn: sqlite3.Connection) -> bool:
    return any(
        conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone()
        for table in ("drafts", "publications")
    )


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the version tables and migrate saved protocols (once, with a backup)."""
    applied = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='version_schema'"
    ).fetchone()
    if applied is not None:
        return
    if _has_saved_records(conn):
        backup_database(conn, "pre-versioning")
    conn.executescript("BEGIN IMMEDIATE;" + _SCHEMA)
    _backfill(conn)
    conn.execute("CREATE TABLE IF NOT EXISTS version_schema (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO version_schema (version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()


def _backfill(conn: sqlite3.Connection) -> None:
    """Give every existing saved protocol a study and version 1.0 (data untouched)."""
    publications = {
        r["protocol_id"]: r for r in conn.execute("SELECT * FROM publications")
    }
    known = set()
    for row in conn.execute("SELECT * FROM drafts ORDER BY updated_at, rowid"):
        known.add(row["protocol_id"])
        _legacy_version(conn, row["protocol_id"], row["name"], row["uploaded_name"],
                        row["upload_fingerprint"], publications.get(row["protocol_id"]))
    for protocol_id, pub in publications.items():
        if protocol_id not in known:
            _legacy_version(conn, protocol_id, pub["name"], None, None, pub)


def _legacy_version(conn, protocol_id, name, uploaded_name, fingerprint, publication) -> None:
    study_id = f"study-{protocol_id}"
    conn.execute(
        "INSERT OR IGNORE INTO studies (study_id, title, created_at) VALUES (?, ?, ?)",
        (study_id, name or protocol_id, _now()),
    )
    is_current = 1 if publication is not None and publication["status"] == "published" else 0
    conn.execute(
        """
        INSERT OR IGNORE INTO protocol_versions
            (protocol_id, study_id, version_label, pdf_status, pdf_name, pdf_sha256,
             published_at, is_current)
        VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
        """,
        (
            protocol_id, study_id, DEFAULT_LABEL, PDF_UNAVAILABLE, uploaded_name,
            publication["published_at"] if publication is not None else None,
            is_current,
        ),
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def validate_label(label: str) -> str:
    cleaned = (label or "").strip()
    if not _LABEL_RE.match(cleaned):
        raise VersionError(f"Invalid version identifier. {LABEL_RULES}")
    return cleaned


def label_taken(study_id: str, label: str, *, conn: Optional[sqlite3.Connection] = None) -> bool:
    own = conn is None
    conn = conn or _connect()
    try:
        return conn.execute(
            "SELECT 1 FROM protocol_versions WHERE study_id = ? AND version_label = ?",
            (study_id, label.strip()),
        ).fetchone() is not None
    finally:
        if own:
            conn.close()


# ---------------------------------------------------------------------------
# Registering versions
# ---------------------------------------------------------------------------

def register_version(
    conn: sqlite3.Connection,
    protocol_id: str,
    *,
    title: str,
    study_id: Optional[str] = None,
    label: Optional[str] = None,
    pdf_name: Optional[str] = None,
) -> None:
    """Attach a protocol_id to a study as a new version. No-op if already registered."""
    if conn.execute(
        "SELECT 1 FROM protocol_versions WHERE protocol_id = ?", (protocol_id,)
    ).fetchone():
        return
    label = validate_label(label or DEFAULT_LABEL)
    if study_id is None:
        study_id = f"study-{uuid.uuid4().hex[:12]}"
        conn.execute(
            "INSERT INTO studies (study_id, title, created_at) VALUES (?, ?, ?)",
            (study_id, title or protocol_id, _now()),
        )
    elif conn.execute(
        "SELECT 1 FROM studies WHERE study_id = ?", (study_id,)
    ).fetchone() is None:
        raise VersionError("That study does not exist.")
    if label_taken(study_id, label, conn=conn):
        raise VersionError(f"Version {label} already exists for this study. Choose a different identifier.")
    conn.execute(
        """
        INSERT INTO protocol_versions
            (protocol_id, study_id, version_label, pdf_status, pdf_name, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (protocol_id, study_id, label, PDF_UNAVAILABLE, pdf_name, _now()),
    )


def promote(conn: sqlite3.Connection, protocol_id: str, title: str) -> None:
    """
    Make this version the study's one current published version.

    Runs inside the caller's publish transaction, so the swap is atomic: the
    previous current version is superseded (its snapshot is kept) and the
    unique index guarantees a single current version.
    """
    register_version(conn, protocol_id, title=title)
    study_id = conn.execute(
        "SELECT study_id FROM protocol_versions WHERE protocol_id = ?", (protocol_id,)
    ).fetchone()["study_id"]
    now = _now()
    others = [
        r["protocol_id"]
        for r in conn.execute(
            "SELECT protocol_id FROM protocol_versions WHERE study_id = ? "
            "AND protocol_id != ? AND is_current = 1",
            (study_id, protocol_id),
        )
    ]
    for other in others:
        conn.execute(
            "UPDATE publications SET status = ? WHERE protocol_id = ? AND status = 'published'",
            (SUPERSEDED, other),
        )
        conn.execute(
            "UPDATE protocol_versions SET is_current = 0, superseded_at = ? WHERE protocol_id = ?",
            (now, other),
        )
    conn.execute(
        "UPDATE protocol_versions SET is_current = 1, published_at = ?, superseded_at = NULL "
        "WHERE protocol_id = ?",
        (now, protocol_id),
    )


def demote(conn: sqlite3.Connection, protocol_id: str) -> None:
    conn.execute(
        "UPDATE protocol_versions SET is_current = 0 WHERE protocol_id = ?", (protocol_id,)
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def get_version(protocol_id: str) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT v.*, s.title AS study_title FROM protocol_versions v "
            "JOIN studies s USING (study_id) WHERE v.protocol_id = ?",
            (protocol_id,),
        ).fetchone()
    return dict(row) if row else None


@_coordinator_only
def previous_version(protocol_id: str) -> Optional[Dict[str, Any]]:
    """The version this one amends: the latest earlier version of the same study."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT rowid AS rid, * FROM protocol_versions WHERE study_id = "
            "(SELECT study_id FROM protocol_versions WHERE protocol_id = ?)",
            (protocol_id,),
        ).fetchall()
    ordered = sorted(rows, key=lambda r: (r["uploaded_at"] or "", r["rid"]))
    ids = [r["protocol_id"] for r in ordered]
    if protocol_id not in ids or ids.index(protocol_id) == 0:
        return None
    return dict(ordered[ids.index(protocol_id) - 1])


def study_title(study_id: str) -> Optional[str]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT title FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
    return row["title"] if row else None


def members(study_id: str) -> set[str]:
    """Every protocol_id (version) of a study."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT protocol_id FROM protocol_versions WHERE study_id = ?", (study_id,)
        ).fetchall()
    return {r["protocol_id"] for r in rows}


def sibling_ids(protocol_id: str) -> set[str]:
    """Other versions of the same study."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT protocol_id FROM protocol_versions WHERE protocol_id != ? AND study_id = "
            "(SELECT study_id FROM protocol_versions WHERE protocol_id = ?)",
            (protocol_id, protocol_id),
        ).fetchall()
    return {r["protocol_id"] for r in rows}


@_coordinator_only
def list_studies() -> List[Dict[str, Any]]:
    """Studies (newest activity first) with their versions, newest first."""
    with closing(_connect()) as conn:
        studies = conn.execute("SELECT * FROM studies").fetchall()
        versions = conn.execute(
            "SELECT v.*, d.updated_at AS saved_at, p.status AS publication "
            "FROM protocol_versions v "
            "LEFT JOIN drafts d USING (protocol_id) "
            "LEFT JOIN publications p USING (protocol_id)"
        ).fetchall()
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in versions:
        grouped.setdefault(row["study_id"], []).append(dict(row))
    out = []
    for study in studies:
        items = sorted(
            grouped.get(study["study_id"], []),
            key=lambda v: (v["uploaded_at"] or v["saved_at"] or ""),
            reverse=True,
        )
        if not items:
            continue
        out.append(
            {
                "study_id": study["study_id"],
                "title": study["title"],
                "versions": items,
                "current": next((v for v in items if v["is_current"]), None),
                "latest_saved": max((v["saved_at"] or "" for v in items), default=""),
            }
        )
    out.sort(key=lambda s: s["latest_saved"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Original PDFs
# ---------------------------------------------------------------------------

def pdf_dir() -> Path:
    from services import review_store

    return Path(os.getenv("CTAI_PDF_DIR") or review_store.db_path().parent / "protocol_pdfs")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_once(path: Path, data: bytes) -> None:
    """Atomic write that refuses to replace a different existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _sha256(path.read_bytes()) == _sha256(data):
            return
        raise VersionError("A different original PDF is already stored for this version.")
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    path.chmod(0o600)


@_coordinator_only
def attach_pdf(protocol_id: str, data: bytes, filename: str) -> bool:
    """
    Store the original PDF for a version (once). Returns True when stored now.

    A version that already has a stored PDF is never changed by this call.
    """
    with closing(_connect()) as conn, conn:
        row = conn.execute(
            "SELECT * FROM protocol_versions WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
        if row is None or row["pdf_status"] == PDF_STORED:
            return False
        digest = _sha256(data)
        path = pdf_dir() / row["study_id"] / f"{protocol_id}.pdf"
        _write_once(path, data)
        conn.execute(
            "UPDATE protocol_versions SET pdf_status = ?, pdf_path = ?, pdf_sha256 = ?, "
            "pdf_name = COALESCE(?, pdf_name) WHERE protocol_id = ?",
            (PDF_STORED, str(path), digest, filename or None, protocol_id),
        )
    return True


def _fingerprint_parts(fingerprint: Optional[str]) -> Optional[tuple[str, str]]:
    """(size, 16-char digest) from the upload fingerprint 'name:size:digest'."""
    if not fingerprint:
        return None
    parts = fingerprint.rsplit(":", 2)
    if len(parts) != 3 or not parts[2]:
        return None
    return parts[1], parts[2]


@_coordinator_only
def reattach_pdf(protocol_id: str, data: bytes, filename: str) -> None:
    """
    Reattach the original PDF of a record whose file was never stored.

    The file must match the fingerprint saved with the record. This never
    creates a new version.
    """
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT v.pdf_status, d.upload_fingerprint FROM protocol_versions v "
            "LEFT JOIN drafts d USING (protocol_id) WHERE v.protocol_id = ?",
            (protocol_id,),
        ).fetchone()
    if row is None:
        raise VersionError("That protocol version does not exist.")
    if row["pdf_status"] == PDF_STORED:
        raise VersionError("The original PDF for this version is already stored.")
    expected = _fingerprint_parts(row["upload_fingerprint"])
    if expected is None:
        raise VersionError(
            "This record has no saved fingerprint, so a file cannot be verified as "
            "its original. The PDF stays unavailable."
        )
    if (str(len(data)), _sha256(data)[:16]) != expected:
        raise VersionError(
            "That file does not match the original upload for this version, so it was not attached."
        )
    attach_pdf(protocol_id, data, filename)


@_coordinator_only
def stored_pdf_bytes(protocol_id: str) -> Optional[bytes]:
    """The stored original, only if it still matches its integrity hash."""
    version = get_version(protocol_id)
    if not version or version["pdf_status"] != PDF_STORED or not version["pdf_path"]:
        return None
    try:
        data = Path(version["pdf_path"]).read_bytes()
    except OSError:
        return None
    return data if _sha256(data) == version["pdf_sha256"] else None


# ---------------------------------------------------------------------------
# Comparison storage
# ---------------------------------------------------------------------------

@_coordinator_only
def save_comparison(
    protocol_id: str, previous_id: str, diff: Dict[str, Any], source_verified: bool
) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            INSERT INTO version_comparisons
                (protocol_id, previous_protocol_id, diff_json, source_verified, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(protocol_id) DO UPDATE SET
                previous_protocol_id = excluded.previous_protocol_id,
                diff_json = excluded.diff_json,
                source_verified = excluded.source_verified,
                created_at = excluded.created_at,
                approved_at = NULL, approved_by = NULL
            """,
            (protocol_id, previous_id, json.dumps(diff), int(source_verified), _now()),
        )


@_coordinator_only
def load_comparison(protocol_id: str) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM version_comparisons WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["diff"] = json.loads(result.pop("diff_json"))
    return result


@_coordinator_only
def approve_comparison(protocol_id: str, actor: str = "coordinator") -> bool:
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "UPDATE version_comparisons SET approved_at = ?, approved_by = ? WHERE protocol_id = ?",
            (_now(), actor, protocol_id),
        )
        return cursor.rowcount > 0
