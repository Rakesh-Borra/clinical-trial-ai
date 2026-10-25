"""
Permanent deletion of a protocol version (or its whole study).

Safeguards, all enforced here and not only in the UI:
- a signed-in coordinator is required (outside a Streamlit session the caller
  must name a coordinator account);
- the exact study name must be typed;
- the coordinator's current password is verified by the existing
  authentication system (Argon2 hash, same lockout as sign-in). It is never
  stored or logged;
- database rows are removed in one transaction together with the audit entry,
  and the stored PDF is moved aside first so a failure restores everything.

Audit history (`history`, `summary_audit`) is kept; `deletion_audit` records
who deleted what and when, without the password.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from services import auth, review_store, version_store

SCOPE_VERSION = "version"
SCOPE_STUDY = "study"

_PER_VERSION_TABLES = (
    "generated_items",
    "publish_acknowledgments",
    "coverage_overrides",
    "summaries",
    "version_comparisons",
    "publications",
    "drafts",
    "protocol_versions",
)


class DeletionError(ValueError):
    """The deletion was refused or failed; nothing was changed."""


def _coordinator(actor_email: Optional[str]) -> Dict[str, Any]:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx(suppress_warning=True) is not None:
        from services import authz

        return authz.require_coordinator()
    user = auth.get_user_by_email(actor_email, auth.COORDINATOR) if actor_email else None
    if user is None:
        raise PermissionError("Deleting a protocol requires a coordinator account.")
    return user


def study_name(protocol_id: str) -> Optional[str]:
    """The name the coordinator must type: the study title, else the saved name."""
    version = version_store.get_version(protocol_id)
    if version and version["study_title"]:
        return version["study_title"]
    with closing(review_store._connect()) as conn:
        row = conn.execute(
            "SELECT name FROM drafts WHERE protocol_id = ?", (protocol_id,)
        ).fetchone()
    return row["name"] if row and row["name"] else None


def _targets(conn: sqlite3.Connection, protocol_id: str, scope: str) -> List[sqlite3.Row]:
    row = conn.execute(
        "SELECT * FROM protocol_versions WHERE protocol_id = ?", (protocol_id,)
    ).fetchone()
    if row is None:
        raise DeletionError("This protocol was not found.")
    if scope == SCOPE_STUDY:
        return conn.execute(
            "SELECT * FROM protocol_versions WHERE study_id = ?", (row["study_id"],)
        ).fetchall()
    return [row]


def _stored_pdf(path: Optional[str]) -> Optional[Path]:
    """The PDF file, only if it lies inside the protected PDF directory."""
    if not path:
        return None
    candidate = Path(path)
    root = version_store.pdf_dir().resolve()
    try:
        candidate.resolve().relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def delete_protocol(
    protocol_id: str,
    *,
    scope: str = SCOPE_VERSION,
    typed_name: str,
    password: str,
    actor_email: Optional[str] = None,
) -> Dict[str, Any]:
    if scope not in (SCOPE_VERSION, SCOPE_STUDY):
        raise DeletionError("Choose whether to delete one version or the whole study.")
    user = _coordinator(actor_email)
    expected = study_name(protocol_id)
    if not expected:
        raise DeletionError("This protocol was not found.")
    if typed_name != expected:
        raise DeletionError("The protocol name does not match exactly.")
    if not password:
        raise DeletionError("Enter your account password.")
    try:
        verified = auth.authenticate(user["email"], password, auth.COORDINATOR)
    except auth.AuthError as exc:
        text = str(exc)
        raise DeletionError(
            text if text.startswith("Too many") else "Your password is incorrect."
        ) from None
    if verified["id"] != user["id"]:
        raise DeletionError("Your password is incorrect.")

    moved: List[tuple[Path, Path]] = []
    try:
        with closing(review_store._connect()) as conn:
            targets = _targets(conn, protocol_id, scope)
            ids = [t["protocol_id"] for t in targets]
            labels = [t["version_label"] for t in targets]
            study_id = targets[0]["study_id"]
            title = conn.execute(
                "SELECT title FROM studies WHERE study_id = ?", (study_id,)
            ).fetchone()["title"]
            was_published = bool(
                conn.execute(
                    "SELECT 1 FROM publications WHERE status = ? AND protocol_id IN (%s)"
                    % ",".join("?" * len(ids)),
                    (review_store.PUBLISHED, *ids),
                ).fetchone()
            )
            # Move PDFs aside first: if the database step fails they are put back.
            for target in targets:
                pdf = _stored_pdf(target["pdf_path"])
                if pdf is not None:
                    aside = pdf.with_name(f"{pdf.name}.{uuid.uuid4().hex}.deleting")
                    os.replace(pdf, aside)
                    moved.append((pdf, aside))
            now = review_store._now()
            try:
                with conn:
                    for pid in ids:
                        for table in _PER_VERSION_TABLES:
                            conn.execute(f"DELETE FROM {table} WHERE protocol_id = ?", (pid,))
                        conn.execute(
                            "INSERT INTO history (protocol_id, at, action, detail) VALUES (?, ?, ?, ?)",
                            (pid, now, "protocol_deleted", f"{scope} deletion by {user['email']}"),
                        )
                    conn.execute(
                        "DELETE FROM studies WHERE study_id = ? AND NOT EXISTS "
                        "(SELECT 1 FROM protocol_versions WHERE study_id = ?)",
                        (study_id, study_id),
                    )
                    conn.execute(
                        "INSERT INTO deletion_audit (at, user_id, user_email, scope, study_id, "
                        "study_title, protocol_ids_json, version_labels_json, was_published, "
                        "pdf_files_removed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            now, user["id"], user["email"], scope, study_id, title,
                            json.dumps(ids), json.dumps(labels), int(was_published), len(moved),
                        ),
                    )
            except BaseException:
                for original, aside in moved:
                    os.replace(aside, original)
                raise
    except DeletionError:
        raise
    except (sqlite3.Error, OSError) as exc:
        raise DeletionError(f"Nothing was deleted: {exc}") from exc

    leftover = 0
    for _, aside in moved:
        try:
            aside.unlink()
        except OSError:
            leftover += 1
    return {
        "deleted_protocol_ids": ids,
        "version_labels": labels,
        "was_published": was_published,
        "pdf_cleanup_incomplete": leftover,
    }


def clear_session_after_delete(session: Any, protocol_ids: List[str]) -> None:
    """
    Remove deleted protocol ids from Streamlit session caches.

    Deletion already removed database rows; this keeps published_protocols and
    coordinator_protocols from treating a deleted study as an active name
    conflict until the next full sync/restart.
    """
    ids = {str(pid) for pid in protocol_ids if pid}
    if not ids:
        return
    published = dict(session.get("published_protocols") or {})
    protocols = dict(session.get("coordinator_protocols") or {})
    for pid in ids:
        published.pop(pid, None)
        protocols.pop(pid, None)
    session["published_protocols"] = published
    session["coordinator_protocols"] = protocols
    if session.get("patient_active_protocol_id") in ids:
        session["patient_active_protocol_id"] = None
        session["patient_history"] = []
