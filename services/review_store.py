"""
SQLite persistence for the coordinator review workflow.

Three small tables, all written with upserts so no action can create a
duplicate record:

- drafts:       the working guide (edits, approvals, rejections) plus the
                protocol analysis it came from, one row per protocol
- publications: the patient-visible snapshot, with a published/withdrawn status
- history:      an append-only log of coordinator actions

The database lives at data/ctai.db unless CTAI_DB_PATH points elsewhere.
Writes made from a live app session require a signed-in coordinator
(see _coordinator_only); bare calls from scripts and tests are unrestricted.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import functools
from typing import Any, Dict, List, Optional, Sequence

from models.plain_language_models import GuideItem, PatientGuide

PUBLISHED = "published"
WITHDRAWN = "withdrawn"
SUPERSEDED = "superseded"

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ctai.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    protocol_id TEXT PRIMARY KEY,
    name TEXT,
    uploaded_name TEXT,
    upload_fingerprint TEXT,
    pipeline_json TEXT,
    guide_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS publications (
    protocol_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    guide_json TEXT NOT NULL,
    status TEXT NOT NULL,
    published_at TEXT,
    withdrawn_at TEXT
);
CREATE TABLE IF NOT EXISTS generated_items (
    protocol_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    item_id TEXT NOT NULL,
    item_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (protocol_id, fingerprint, item_id)
);
CREATE TABLE IF NOT EXISTS summaries (
    protocol_id TEXT PRIMARY KEY,
    original_text TEXT NOT NULL,
    generated_by TEXT NOT NULL,
    draft_text TEXT NOT NULL,
    approved_text TEXT,
    approved_at TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summary_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_id TEXT NOT NULL,
    at TEXT NOT NULL,
    action TEXT NOT NULL,
    before_text TEXT,
    after_text TEXT,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS coverage_overrides (
    protocol_id TEXT NOT NULL,
    category TEXT NOT NULL,
    reason TEXT NOT NULL,
    at TEXT NOT NULL,
    PRIMARY KEY (protocol_id, category)
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_id TEXT NOT NULL,
    at TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS publish_acknowledgments (
    protocol_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    missing_json TEXT NOT NULL,
    user_id TEXT,
    user_email TEXT,
    version_label TEXT,
    at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deletion_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    user_id TEXT,
    user_email TEXT,
    scope TEXT NOT NULL,
    study_id TEXT,
    study_title TEXT,
    protocol_ids_json TEXT NOT NULL,
    version_labels_json TEXT NOT NULL,
    was_published INTEGER NOT NULL DEFAULT 0,
    pdf_files_removed INTEGER NOT NULL DEFAULT 0
);
"""


def _coordinator_only(func):
    """Refuse writes from a live Streamlit session unless a coordinator is signed in."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is not None:
            from services import authz

            authz.require_coordinator()
        return func(*args, **kwargs)

    return wrapper


def db_path() -> Path:
    return Path(os.getenv("CTAI_DB_PATH") or DEFAULT_DB_PATH)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    from services import version_store

    version_store.ensure_schema(conn)
    return conn


def _run(sql: str, params: tuple = ()) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(sql, params)


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------

@_coordinator_only
def save_draft(
    protocol_id: str,
    guide: PatientGuide,
    *,
    name: str = "",
    uploaded_name: Optional[str] = None,
    upload_fingerprint: Optional[str] = None,
    pipeline: Optional[Dict[str, Any]] = None,
    study_id: Optional[str] = None,
    version_label: Optional[str] = None,
) -> None:
    """
    Insert or replace the working guide for one protocol.

    The first save also registers the protocol as a version of a study (a new
    study unless ``study_id`` is given). A duplicate version label in that
    study raises VersionError and nothing is saved.
    """
    from services import version_store

    pipeline_json = None
    if pipeline is not None:
        try:
            pipeline_json = json.dumps(_pipeline_to_jsonable(pipeline))
        except (TypeError, ValueError):
            pipeline_json = None
    with closing(_connect()) as conn, conn:
        existing = conn.execute(
            "SELECT pipeline_json FROM drafts WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
        if pipeline_json is None and existing is not None:
            pipeline_json = existing["pipeline_json"]
        version_store.register_version(
            conn, protocol_id, title=name, study_id=study_id,
            label=version_label, pdf_name=uploaded_name,
        )
        conn.execute(
            """
            INSERT INTO drafts (protocol_id, name, uploaded_name, upload_fingerprint,
                                pipeline_json, guide_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(protocol_id) DO UPDATE SET
                name = excluded.name,
                uploaded_name = excluded.uploaded_name,
                upload_fingerprint = excluded.upload_fingerprint,
                pipeline_json = excluded.pipeline_json,
                guide_json = excluded.guide_json,
                updated_at = excluded.updated_at
            """,
            (
                protocol_id,
                name,
                uploaded_name,
                upload_fingerprint,
                pipeline_json,
                guide.model_dump_json(),
                _now(),
            ),
        )


def save_generated_item(protocol_id: str, fingerprint: str, item: GuideItem) -> None:
    """Persist one finished statement the moment it is generated (no decisions)."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            INSERT INTO generated_items (protocol_id, fingerprint, item_id, item_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(protocol_id, fingerprint, item_id) DO UPDATE SET
                item_json = excluded.item_json, updated_at = excluded.updated_at
            """,
            (protocol_id, fingerprint, item.item_id, item.model_dump_json(), _now()),
        )


def load_generated_items(protocol_id: str, fingerprint: str) -> Dict[str, GuideItem]:
    """Statements generated so far for this analysis, by stable id."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT item_id, item_json FROM generated_items "
            "WHERE protocol_id = ? AND fingerprint = ?",
            (protocol_id, fingerprint),
        ).fetchall()
    out: Dict[str, GuideItem] = {}
    for row in rows:
        try:
            out[row["item_id"]] = GuideItem.model_validate_json(row["item_json"])
        except ValueError:
            continue  # an unreadable row is regenerated, not trusted
    return out


@_coordinator_only
def clear_generated_items(protocol_id: str, keep_fingerprint: Optional[str] = None) -> None:
    """Drop stored generation output (other analyses of the same protocol)."""
    with closing(_connect()) as conn, conn:
        conn.execute(
            "DELETE FROM generated_items WHERE protocol_id = ? AND fingerprint != ?",
            (protocol_id, keep_fingerprint or ""),
        )


def _draft_record(row: sqlite3.Row) -> Optional[Dict[str, Any]]:
    """A saved draft as a dict, or None when its guide cannot be read."""
    try:
        guide = PatientGuide.model_validate_json(row["guide_json"])
    except ValueError:
        return None
    return {
        "protocol_id": row["protocol_id"],
        "name": row["name"] or "",
        "uploaded_name": row["uploaded_name"],
        "upload_fingerprint": row["upload_fingerprint"],
        "pipeline": _pipeline_from_json(row["pipeline_json"]),
        "guide": guide,
        "updated_at": row["updated_at"],
    }


def load_latest_draft() -> Optional[Dict[str, Any]]:
    """The most recently saved draft, or None. Unreadable rows are skipped."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM drafts ORDER BY updated_at DESC, rowid DESC"
        ).fetchall()
    for row in rows:
        record = _draft_record(row)
        if record is not None:
            return record
    return None


@_coordinator_only
def load_draft(protocol_id: str) -> Optional[Dict[str, Any]]:
    """One saved draft by id, or None when it is missing or unreadable."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM drafts WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
    return _draft_record(row) if row else None


@_coordinator_only
def list_protocols() -> List[Dict[str, Any]]:
    """
    Every saved protocol, newest first, with only what is actually stored.

    Each entry has the draft's name, original filename and last-saved time,
    whether its analysis was saved, its statement guide (None if unreadable),
    the publication status, whether the summary was approved, and its study
    and version details.
    """
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM drafts ORDER BY updated_at DESC, rowid DESC"
        ).fetchall()
        published = {
            r["protocol_id"]: r["status"]
            for r in conn.execute("SELECT protocol_id, status FROM publications")
        }
        approved = {
            r["protocol_id"]
            for r in conn.execute(
                "SELECT protocol_id FROM summaries WHERE approved_text IS NOT NULL"
            )
        }
        versions = {
            r["protocol_id"]: r
            for r in conn.execute(
                "SELECT v.*, s.title AS study_title FROM protocol_versions v "
                "JOIN studies s USING (study_id)"
            )
        }
    out: List[Dict[str, Any]] = []
    for row in rows:
        record = _draft_record(row)
        version = versions.get(row["protocol_id"])
        out.append(
            {
                "study_id": version["study_id"] if version else None,
                "study_title": version["study_title"] if version else None,
                "version_label": version["version_label"] if version else None,
                "is_current": bool(version and version["is_current"]),
                "published_at": version["published_at"] if version else None,
                "uploaded_at": version["uploaded_at"] if version else None,
                "pdf_status": version["pdf_status"] if version else None,
                "protocol_id": row["protocol_id"],
                "name": row["name"] or "",
                "uploaded_name": row["uploaded_name"],
                "updated_at": row["updated_at"],
                "has_analysis": bool(record and record["pipeline"] is not None),
                "guide": record["guide"] if record else None,
                "publication": published.get(row["protocol_id"]),
                "summary_approved": row["protocol_id"] in approved,
            }
        )
    return out


def draft_exists(protocol_id: str) -> bool:
    """True when a draft row has been saved for this protocol."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT 1 FROM drafts WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
    return row is not None


def draft_names() -> Dict[str, str]:
    """protocol_id -> saved name for every draft (used for duplicate checks)."""
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT protocol_id, name FROM drafts").fetchall()
    return {r["protocol_id"]: r["name"] for r in rows if r["name"]}


def _pipeline_to_jsonable(pipeline: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(pipeline)
    analysis = data.get("analysis")
    if hasattr(analysis, "model_dump"):
        data["analysis"] = analysis.model_dump(mode="json")
    return data


def _pipeline_from_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    from models.protocol_models import ProtocolAnalysis

    try:
        data = json.loads(raw)
        data["analysis"] = ProtocolAnalysis.model_validate(data["analysis"])
    except (ValueError, KeyError, TypeError):
        return None
    return data


# ---------------------------------------------------------------------------
# Publications
# ---------------------------------------------------------------------------

class AcknowledgmentError(ValueError):
    """Publishing with missing categories needs a current coordinator acknowledgment."""


def _signed_in_identity() -> tuple[Optional[str], Optional[str]]:
    from services import authz

    user = authz.current_user() or {}
    return (str(user["id"]) if user.get("id") is not None else None), user.get("email")


@_coordinator_only
def record_incomplete_acknowledgment(
    protocol_id: str,
    fingerprint: str,
    missing: Sequence[str],
    version_label: Optional[str] = None,
) -> None:
    """Remember that the signed-in coordinator accepted publishing without these categories."""
    user_id, email = _signed_in_identity()
    at = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO publish_acknowledgments (protocol_id, fingerprint, missing_json, "
            "user_id, user_email, version_label, at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(protocol_id) DO UPDATE SET fingerprint = excluded.fingerprint, "
            "missing_json = excluded.missing_json, user_id = excluded.user_id, "
            "user_email = excluded.user_email, version_label = excluded.version_label, "
            "at = excluded.at",
            (protocol_id, fingerprint, json.dumps(list(missing)), user_id, email, version_label, at),
        )
        conn.execute(
            "INSERT INTO history (protocol_id, at, action, detail) VALUES (?, ?, ?, ?)",
            (
                protocol_id, at, "incomplete_publication_acknowledged",
                _ack_detail(email, user_id, protocol_id, version_label, missing, at),
            ),
        )


@_coordinator_only
def clear_incomplete_acknowledgment(protocol_id: str) -> None:
    _run("DELETE FROM publish_acknowledgments WHERE protocol_id = ?", (protocol_id,))


def load_incomplete_acknowledgment(protocol_id: str) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM publish_acknowledgments WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
    return dict(row) if row else None


def acknowledgment_valid(protocol_id: str, fingerprint: str) -> bool:
    """True when a stored acknowledgment covers exactly this content, version and gap list."""
    row = load_incomplete_acknowledgment(protocol_id)
    return bool(row) and row["fingerprint"] == fingerprint


def _ack_detail(email, user_id, protocol_id, version_label, missing, at) -> str:
    return (
        f"by {email or 'unknown'} (user {user_id or '?'}) | protocol {protocol_id} | "
        f"version {version_label or '?'} | missing: {'; '.join(missing)} | acknowledged {at}"
    )


@_coordinator_only
def publish(
    protocol_id: str,
    name: str,
    snapshot: PatientGuide,
    missing_categories: Sequence[str] = (),
    acknowledgment_fingerprint: Optional[str] = None,
) -> None:
    """
    Make a snapshot patient-visible and the study's one current version.

    The previous current version of the same study is superseded in the same
    transaction; its snapshot is kept as history. When critical categories are
    missing, a stored acknowledgment for exactly this content is required and
    is written to the audit log with the publication.
    """
    from services import version_store

    with closing(_connect()) as conn, conn:
        if missing_categories:
            ack = conn.execute(
                "SELECT * FROM publish_acknowledgments WHERE protocol_id = ?", (protocol_id,)
            ).fetchone()
            if (
                ack is None
                or not acknowledgment_fingerprint
                or ack["fingerprint"] != acknowledgment_fingerprint
                or json.loads(ack["missing_json"]) != list(missing_categories)
            ):
                raise AcknowledgmentError(
                    "Missing categories must be acknowledged for this version before publishing."
                )
            conn.execute(
                "INSERT INTO history (protocol_id, at, action, detail) VALUES (?, ?, ?, ?)",
                (
                    protocol_id, _now(), "published_with_missing_categories",
                    _ack_detail(
                        ack["user_email"], ack["user_id"], protocol_id,
                        ack["version_label"], missing_categories, ack["at"],
                    ),
                ),
            )
        conn.execute(
            """
            INSERT INTO publications (protocol_id, name, guide_json, status,
                                      published_at, withdrawn_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            ON CONFLICT(protocol_id) DO UPDATE SET
                name = excluded.name,
                guide_json = excluded.guide_json,
                status = excluded.status,
                published_at = excluded.published_at,
                withdrawn_at = NULL
            """,
            (protocol_id, name, snapshot.model_dump_json(), PUBLISHED, _now()),
        )
        version_store.promote(conn, protocol_id, name)


@_coordinator_only
def unpublish(protocol_id: str) -> bool:
    """Withdraw a publication. Returns True when something was withdrawn."""
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "UPDATE publications SET status = ?, withdrawn_at = ? "
            "WHERE protocol_id = ? AND status = ?",
            (WITHDRAWN, _now(), protocol_id, PUBLISHED),
        )
        if cursor.rowcount > 0:
            from services import version_store

            version_store.demote(conn, protocol_id)
        return cursor.rowcount > 0


def publication_status(protocol_id: str) -> Optional[str]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT status FROM publications WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
    return row["status"] if row else None


def load_publications() -> Dict[str, Dict[str, Any]]:
    """
    protocol_id -> {"name", "guide", "status"} for every stored publication.

    Rows whose snapshot cannot be read are omitted, so they can never reach
    patients.
    """
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT p.*, v.version_label, v.published_at AS version_published_at "
            "FROM publications p LEFT JOIN protocol_versions v USING (protocol_id)"
        ).fetchall()
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            guide = PatientGuide.model_validate_json(row["guide_json"])
        except ValueError:
            continue
        result[row["protocol_id"]] = {
            "name": row["name"],
            "guide": guide,
            "status": row["status"],
            "version_label": row["version_label"],
            "published_at": row["published_at"] or row["version_published_at"],
        }
    return result


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@_coordinator_only
def log_action(protocol_id: str, action: str, detail: str = "") -> None:
    _run(
        "INSERT INTO history (protocol_id, at, action, detail) VALUES (?, ?, ?, ?)",
        (protocol_id, _now(), action, detail[:500]),
    )


def history(protocol_id: str, limit: int = 20) -> List[Dict[str, str]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT at, action, detail FROM history WHERE protocol_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (protocol_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Patient summary: the AI original is never overwritten by coordinator edits
# ---------------------------------------------------------------------------

def _audit_summary(conn, protocol_id, action, before=None, after=None, detail=""):
    conn.execute(
        "INSERT INTO summary_audit (protocol_id, at, action, before_text, "
        "after_text, detail) VALUES (?, ?, ?, ?, ?, ?)",
        (protocol_id, _now(), action, before, after, detail[:500]),
    )


def load_summary(protocol_id: str) -> Optional[Dict[str, Any]]:
    """The saved summary record, or None. Approval holds only while unchanged."""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM summaries WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
    return dict(row) if row else None


@_coordinator_only
def save_original_summary(protocol_id: str, text: str, generated_by: str) -> None:
    """Store a newly generated AI summary; it also becomes the working draft."""
    with closing(_connect()) as conn, conn:
        previous = conn.execute(
            "SELECT draft_text FROM summaries WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
        conn.execute(
            """
            INSERT INTO summaries (protocol_id, original_text, generated_by,
                                   draft_text, approved_text, approved_at, updated_at)
            VALUES (?, ?, ?, ?, NULL, NULL, ?)
            ON CONFLICT(protocol_id) DO UPDATE SET
                original_text = excluded.original_text,
                generated_by = excluded.generated_by,
                draft_text = excluded.draft_text,
                approved_text = NULL,
                approved_at = NULL,
                updated_at = excluded.updated_at
            """,
            (protocol_id, text, generated_by, text, _now()),
        )
        _audit_summary(
            conn, protocol_id, "generated",
            previous["draft_text"] if previous else None, text, generated_by,
        )


@_coordinator_only
def save_summary_draft(protocol_id: str, text: str, actor: str = "coordinator") -> bool:
    """
    Save a coordinator edit as the draft; the original is untouched.

    Any earlier approval is withdrawn until the new wording is approved.
    Returns False when no summary exists or nothing changed.
    """
    with closing(_connect()) as conn, conn:
        row = conn.execute(
            "SELECT draft_text, approved_text FROM summaries WHERE protocol_id = ?",
            (protocol_id,),
        ).fetchone()
        if row is None or row["draft_text"] == text:
            return False
        conn.execute(
            "UPDATE summaries SET draft_text = ?, approved_text = NULL, "
            "approved_at = NULL, updated_at = ? WHERE protocol_id = ?",
            (text, _now(), protocol_id),
        )
        _audit_summary(
            conn, protocol_id, "draft_saved", row["draft_text"], text, actor
        )
        if row["approved_text"] is not None:
            _audit_summary(
                conn, protocol_id, "approval_invalidated",
                row["approved_text"], text, "edited after approval",
            )
    return True


@_coordinator_only
def approve_summary(protocol_id: str, actor: str = "coordinator") -> bool:
    """Approve exactly the current draft text."""
    with closing(_connect()) as conn, conn:
        row = conn.execute(
            "SELECT draft_text FROM summaries WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE summaries SET approved_text = draft_text, approved_at = ?, "
            "updated_at = ? WHERE protocol_id = ?",
            (_now(), _now(), protocol_id),
        )
        _audit_summary(
            conn, protocol_id, "approved", None, row["draft_text"], actor
        )
    return True


@_coordinator_only
def withdraw_summary_approval(protocol_id: str) -> None:
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "UPDATE summaries SET approved_text = NULL, approved_at = NULL "
            "WHERE protocol_id = ? AND approved_text IS NOT NULL",
            (protocol_id,),
        )
        if cursor.rowcount:
            _audit_summary(conn, protocol_id, "approval_withdrawn")


def summary_audit(protocol_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT at, action, before_text, after_text, detail FROM summary_audit "
            "WHERE protocol_id = ? ORDER BY id DESC LIMIT ?",
            (protocol_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# "Not applicable" overrides for critical categories
# ---------------------------------------------------------------------------

@_coordinator_only
def set_coverage_override(protocol_id: str, category: str, reason: str) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            INSERT INTO coverage_overrides (protocol_id, category, reason, at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(protocol_id, category) DO UPDATE SET
                reason = excluded.reason, at = excluded.at
            """,
            (protocol_id, category, reason, _now()),
        )
    log_action(protocol_id, "category_not_applicable", f"{category}: {reason}")


@_coordinator_only
def clear_coverage_override(protocol_id: str, category: str) -> None:
    _run(
        "DELETE FROM coverage_overrides WHERE protocol_id = ? AND category = ?",
        (protocol_id, category),
    )
    log_action(protocol_id, "category_override_removed", category)


def load_coverage_overrides(protocol_id: str) -> Dict[str, Dict[str, str]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT category, reason, at FROM coverage_overrides WHERE protocol_id = ?",
            (protocol_id,),
        ).fetchall()
    return {r["category"]: {"reason": r["reason"], "at": r["at"]} for r in rows}
