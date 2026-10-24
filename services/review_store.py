"""
SQLite persistence for the coordinator review workflow.

Three small tables, all written with upserts so no action can create a
duplicate record:

- drafts:       the working guide (edits, approvals, rejections) plus the
                protocol analysis it came from, one row per protocol
- publications: the patient-visible snapshot, with a published/withdrawn status
- history:      an append-only log of coordinator actions

The database lives at data/ctai.db unless CTAI_DB_PATH points elsewhere.
Nothing here imports Streamlit, so the rules are easy to test.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.plain_language_models import PatientGuide

PUBLISHED = "published"
WITHDRAWN = "withdrawn"

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
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_id TEXT NOT NULL,
    at TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT
);
"""


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
    return conn


def _run(sql: str, params: tuple = ()) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(sql, params)


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------

def save_draft(
    protocol_id: str,
    guide: PatientGuide,
    *,
    name: str = "",
    uploaded_name: Optional[str] = None,
    upload_fingerprint: Optional[str] = None,
    pipeline: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert or replace the working guide for one protocol."""
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


def load_latest_draft() -> Optional[Dict[str, Any]]:
    """The most recently saved draft, or None. Unreadable rows are skipped."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM drafts ORDER BY updated_at DESC, rowid DESC"
        ).fetchall()
    for row in rows:
        try:
            guide = PatientGuide.model_validate_json(row["guide_json"])
        except ValueError:
            continue
        return {
            "protocol_id": row["protocol_id"],
            "name": row["name"] or "",
            "uploaded_name": row["uploaded_name"],
            "upload_fingerprint": row["upload_fingerprint"],
            "pipeline": _pipeline_from_json(row["pipeline_json"]),
            "guide": guide,
        }
    return None


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

def publish(protocol_id: str, name: str, snapshot: PatientGuide) -> None:
    """Make a snapshot patient-visible (replaces any earlier version)."""
    with closing(_connect()) as conn, conn:
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


def unpublish(protocol_id: str) -> bool:
    """Withdraw a publication. Returns True when something was withdrawn."""
    with closing(_connect()) as conn, conn:
        cursor = conn.execute(
            "UPDATE publications SET status = ?, withdrawn_at = ? "
            "WHERE protocol_id = ? AND status = ?",
            (WITHDRAWN, _now(), protocol_id, PUBLISHED),
        )
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
        rows = conn.execute("SELECT * FROM publications").fetchall()
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
        }
    return result


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

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
