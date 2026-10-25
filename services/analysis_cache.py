"""
Cache of completed model analyses, stored in the existing SQLite database.

The key hashes the exact text sent to the model plus every setting that can
change the result (provider, model, chunking, routing, think/num_predict,
prompt version), so changing the document or any setting is a cache miss.
Only complete analyses without failures are stored, so a partial result is
never replayed. Source verification and accuracy checks always run afresh on
the original PDF; only the model output is reused.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from models.protocol_models import ProtocolAnalysis
from services.review_store import db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_cache (
    cache_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def make_key(text: str, settings: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.executescript(_SCHEMA)
    return conn


def load(key: str) -> Optional[Dict[str, Any]]:
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                "SELECT payload FROM analysis_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        data["analysis"] = ProtocolAnalysis.model_validate(data["analysis"])
        return data
    except Exception:
        return None  # a broken cache must never break analysis


def store(key: str, analysis: ProtocolAnalysis, warnings: list[str]) -> None:
    payload = json.dumps({
        "analysis": analysis.model_dump(mode="json"),
        "warnings": list(warnings),
    })
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO analysis_cache VALUES (?, ?, ?)",
                (key, payload,
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
    except Exception:
        pass
