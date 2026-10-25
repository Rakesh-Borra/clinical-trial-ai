"""
Accounts, sessions and password recovery, stored in the application's SQLite file.

* Passwords are hashed with Argon2id; plain passwords are never stored or logged.
* Public registration can only create ``patient`` accounts. Coordinators are
  created by an administrator with ``scripts/manage_users.py``.
* Session and reset tokens are random; only their SHA-256 digest is stored.
* Reset tokens are single use and expire. They are delivered only by email
  (SMTP settings come from environment variables) and are never displayed or logged.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import smtplib
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Any, Dict, Optional

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from services import review_store

PATIENT = "patient"
COORDINATOR = "coordinator"
ROLES = (PATIENT, COORDINATOR)

SESSION_HOURS = 8
RESET_MINUTES = 30
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15
MIN_PASSWORD_LENGTH = 10

PASSWORD_GUIDANCE = (
    f"Use at least {MIN_PASSWORD_LENGTH} characters with letters and numbers. "
    "A longer phrase is stronger."
)
GENERIC_LOGIN_ERROR = "Incorrect email or password."
GENERIC_RESET_NOTICE = (
    "If an account exists for that email, a reset link has been sent. "
    "The link expires in 30 minutes and works once."
)

_hasher = PasswordHasher()
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_MIGRATIONS = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('patient', 'coordinator')),
        created_at TEXT NOT NULL,
        reset_token_hash TEXT,
        reset_expires_at TEXT
    );
    CREATE TABLE IF NOT EXISTS auth_sessions (
        token_hash TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS login_attempts (
        email TEXT NOT NULL,
        attempted_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_login_attempts_email ON login_attempts(email);
    """,
]


class AuthError(Exception):
    """A user-facing authentication problem."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    path = review_store.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE TABLE IF NOT EXISTS auth_schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT MAX(version) AS v FROM auth_schema_version").fetchone()
    applied = row["v"] or 0
    for number, script in enumerate(_MIGRATIONS, start=1):
        if number > applied:
            conn.executescript(script)
            conn.execute("INSERT INTO auth_schema_version (version) VALUES (?)", (number,))
            conn.commit()
    return conn


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_password(password: str, email: str = "") -> Optional[str]:
    """Return a friendly problem description, or None when the password is acceptable."""
    if len(password or "") < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Password must include both letters and numbers."
    if email and password.strip().lower() == normalize_email(email):
        return "Password must not be the same as your email."
    return None


def _public(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "role": row["role"],
        "created_at": row["created_at"],
    }


def create_user(full_name: str, email: str, password: str, role: str = PATIENT) -> Dict[str, Any]:
    """Create an account. Administrators call this directly; the UI uses register_patient."""
    name = (full_name or "").strip()
    address = normalize_email(email)
    if role not in ROLES:
        raise AuthError("Unknown role.")
    if not name:
        raise AuthError("Enter your full name.")
    if not _EMAIL.match(address):
        raise AuthError("Enter a valid email address.")
    problem = validate_password(password, address)
    if problem:
        raise AuthError(problem)
    user_id = uuid.uuid4().hex
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO users (id, full_name, email, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, name, address, _hasher.hash(password), role, _iso(_now())),
            )
    except sqlite3.IntegrityError:
        raise AuthError("An account with this email already exists.") from None
    return {"id": user_id, "full_name": name, "email": address, "role": role}


def register_patient(full_name: str, email: str, password: str, confirm: str) -> Dict[str, Any]:
    """Public registration. The role is always patient; callers cannot choose it."""
    if password != confirm:
        raise AuthError("Passwords do not match.")
    return create_user(full_name, email, password, PATIENT)


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (normalize_email(email),)
        ).fetchone()
    return _public(row) if row else None


# ---------------------------------------------------------------------------
# Login, sessions and rate limiting
# ---------------------------------------------------------------------------

def _recent_failures(conn: sqlite3.Connection, email: str) -> int:
    since = _iso(_now() - timedelta(minutes=LOCKOUT_MINUTES))
    return conn.execute(
        "SELECT COUNT(*) FROM login_attempts WHERE email = ? AND attempted_at >= ?",
        (email, since),
    ).fetchone()[0]


def is_locked_out(email: str) -> bool:
    with closing(_connect()) as conn:
        return _recent_failures(conn, normalize_email(email)) >= MAX_FAILED_LOGINS


def authenticate(email: str, password: str) -> Dict[str, Any]:
    """Verify credentials; raises AuthError with a deliberately generic message."""
    address = normalize_email(email)
    if not address or not password:
        raise AuthError("Enter your email and password.")
    with closing(_connect()) as conn, conn:
        if _recent_failures(conn, address) >= MAX_FAILED_LOGINS:
            raise AuthError(
                f"Too many failed attempts. Try again in {LOCKOUT_MINUTES} minutes."
            )
        row = conn.execute("SELECT * FROM users WHERE email = ?", (address,)).fetchone()
        # Unknown emails still pay for one hash check so timing does not reveal them.
        stored = row["password_hash"] if row else _DUMMY_HASH
        try:
            valid = _hasher.verify(stored, password)
        except (VerificationError, InvalidHashError):
            valid = False
        if not (valid and row):
            conn.execute(
                "INSERT INTO login_attempts (email, attempted_at) VALUES (?, ?)",
                (address, _iso(_now())),
            )
            conn.commit()  # keep the failure on record even though we raise
            raise AuthError(GENERIC_LOGIN_ERROR)
        conn.execute("DELETE FROM login_attempts WHERE email = ?", (address,))
        if _hasher.check_needs_rehash(row["password_hash"]):
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (_hasher.hash(password), row["id"]),
            )
        return _public(row)


_DUMMY_HASH = _hasher.hash("not-a-real-password-0")


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (_iso(now),))
        conn.execute(
            "INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (_digest(token), user_id, _iso(now), _iso(now + timedelta(hours=SESSION_HOURS))),
        )
    return token


def user_for_session(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """The signed-in user for a session token, or None if missing, expired or revoked."""
    if not token:
        return None
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT u.* FROM auth_sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = ? AND s.expires_at >= ?",
            (_digest(token), _iso(_now())),
        ).fetchone()
    return _public(row) if row else None


def end_session(token: Optional[str]) -> None:
    if not token:
        return
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_digest(token),))


def _revoke_user_sessions(conn: sqlite3.Connection, user_id: str) -> None:
    conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))


# ---------------------------------------------------------------------------
# Password change and recovery
# ---------------------------------------------------------------------------

def change_password(user_id: str, current: str, new: str, confirm: str) -> None:
    with closing(_connect()) as conn, conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            raise AuthError("Account not found.")
        try:
            _hasher.verify(row["password_hash"], current)
        except (VerificationError, InvalidHashError):
            raise AuthError("Your current password is incorrect.") from None
        if new != confirm:
            raise AuthError("New passwords do not match.")
        problem = validate_password(new, row["email"])
        if problem:
            raise AuthError(problem)
        conn.execute(
            "UPDATE users SET password_hash = ?, reset_token_hash = NULL, "
            "reset_expires_at = NULL WHERE id = ?",
            (_hasher.hash(new), user_id),
        )


def email_delivery_configured() -> bool:
    return bool(os.getenv("CTAI_SMTP_HOST") and os.getenv("CTAI_SMTP_FROM"))


def send_email(to: str, subject: str, body: str) -> None:
    host = os.getenv("CTAI_SMTP_HOST", "")
    port = int(os.getenv("CTAI_SMTP_PORT") or 587)
    message = EmailMessage()
    message["From"] = os.getenv("CTAI_SMTP_FROM", "")
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(host, port, timeout=15) as smtp:
        smtp.starttls()
        user = os.getenv("CTAI_SMTP_USER")
        if user:
            smtp.login(user, os.getenv("CTAI_SMTP_PASSWORD", ""))
        smtp.send_message(message)


def request_password_reset(email: str) -> str:
    """
    Email a single-use reset token when the account exists.

    Always returns the same notice so the response does not reveal whether an
    account exists. Raises AuthError only when email delivery is not configured.
    """
    if not email_delivery_configured():
        raise AuthError(
            "Password recovery is unavailable because email delivery is not "
            "configured. Ask your administrator to reset your password."
        )
    address = normalize_email(email)
    user = get_user_by_email(address)
    if user:
        token = secrets.token_urlsafe(32)
        expires = _iso(_now() + timedelta(minutes=RESET_MINUTES))
        with closing(_connect()) as conn, conn:
            conn.execute(
                "UPDATE users SET reset_token_hash = ?, reset_expires_at = ? WHERE id = ?",
                (_digest(token), expires, user["id"]),
            )
        base = (os.getenv("CTAI_APP_URL") or "").rstrip("/")
        link = f"{base}/?reset_token={token}" if base else ""
        body = (
            "A password reset was requested for your Clinical Trial AI account.\n\n"
            + (f"Open this link to choose a new password:\n{link}\n\n" if link else "")
            + f"Reset code: {token}\n\n"
            f"It expires in {RESET_MINUTES} minutes and can be used once. "
            "If you did not request this, ignore this email."
        )
        try:
            send_email(address, "Reset your Clinical Trial AI password", body)
        except Exception:
            # Delivery failures are not reported per account, to avoid revealing it.
            pass
    return GENERIC_RESET_NOTICE


def reset_password(token: str, new: str, confirm: str) -> None:
    token = (token or "").strip()
    if not token:
        raise AuthError("Enter the reset code from your email.")
    if new != confirm:
        raise AuthError("Passwords do not match.")
    with closing(_connect()) as conn, conn:
        row = conn.execute(
            "SELECT * FROM users WHERE reset_token_hash = ?", (_digest(token),)
        ).fetchone()
        if not row or not row["reset_expires_at"] or row["reset_expires_at"] < _iso(_now()):
            raise AuthError("This reset link is invalid or has expired. Request a new one.")
        problem = validate_password(new, row["email"])
        if problem:
            raise AuthError(problem)
        conn.execute(
            "UPDATE users SET password_hash = ?, reset_token_hash = NULL, "
            "reset_expires_at = NULL WHERE id = ?",
            (_hasher.hash(new), row["id"]),
        )
        _revoke_user_sessions(conn, row["id"])
        conn.execute("DELETE FROM login_attempts WHERE email = ?", (row["email"],))


def admin_set_password(email: str, new: str) -> None:
    """Administrator reset (command line); ends the user's sessions."""
    address = normalize_email(email)
    problem = validate_password(new, address)
    if problem:
        raise AuthError(problem)
    with closing(_connect()) as conn, conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (address,)).fetchone()
        if not row:
            raise AuthError("No account with that email.")
        conn.execute(
            "UPDATE users SET password_hash = ?, reset_token_hash = NULL, "
            "reset_expires_at = NULL WHERE id = ?",
            (_hasher.hash(new), row["id"]),
        )
        _revoke_user_sessions(conn, row["id"])
        conn.execute("DELETE FROM login_attempts WHERE email = ?", (address,))


def set_role(email: str, role: str) -> None:
    """Administrator action (command line only; never exposed in the UI)."""
    if role not in ROLES:
        raise AuthError("Unknown role.")
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "UPDATE users SET role = ? WHERE email = ?", (role, normalize_email(email))
        )
        if cur.rowcount == 0:
            raise AuthError("No account with that email.")
        conn.execute(
            "DELETE FROM auth_sessions WHERE user_id IN (SELECT id FROM users WHERE email = ?)",
            (normalize_email(email),),
        )
