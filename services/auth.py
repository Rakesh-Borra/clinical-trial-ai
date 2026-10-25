"""
Accounts, sessions and password recovery, stored in the application's SQLite file.

* Passwords are hashed with Argon2id; plain passwords are never stored or logged.
* An email address may hold one patient account and one coordinator account;
  (email, role) is unique. Every lookup, login, reset and rate limit is
  per (email, role), so the two accounts never see each other.
* Public registration creates ``patient`` accounts. A ``coordinator`` account can
  only be registered with a single-use, expiring invitation code that an
  administrator issues with ``scripts/manage_users.py create-invite``; only the
  code's hash is stored.
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
GENERIC_LOGIN_ERROR = "Incorrect email, password or account type."
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
    """
    CREATE TABLE IF NOT EXISTS coordinator_invites (
        code_hash TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        used_at TEXT,
        used_by TEXT
    );
    """,
    # Email-only uniqueness becomes (email, role) uniqueness. The table is rebuilt
    # with foreign keys off (see _connect) so sessions are not cascade-deleted.
    """
    BEGIN;
    CREATE TABLE users_new (
        id TEXT PRIMARY KEY,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('patient', 'coordinator')),
        created_at TEXT NOT NULL,
        reset_token_hash TEXT,
        reset_expires_at TEXT,
        UNIQUE (email, role)
    );
    INSERT INTO users_new
        SELECT id, full_name, email, password_hash, role, created_at,
               reset_token_hash, reset_expires_at FROM users;
    DROP TABLE users;
    ALTER TABLE users_new RENAME TO users;
    ALTER TABLE login_attempts ADD COLUMN role TEXT NOT NULL DEFAULT '';
    COMMIT;
    """,
]
INVITE_HOURS = 72
DEMO_MODE_ENV = "CTAI_DEMO_MODE"
DEMO_CODE = "0000"
DEMO_CODE_HINT = 'Enter "0000" as your code'
DEMO_LABEL = "Classroom demo access only, not production security."
INVALID_INVITE = "That invitation code is invalid, expired or already used."


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
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.executescript(script)
            conn.execute("PRAGMA foreign_keys = ON")
            if conn.execute("PRAGMA foreign_key_check").fetchall():
                raise sqlite3.IntegrityError("Authentication migration broke a foreign key.")
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


def create_user(
    full_name: str, email: str, password: str, role: str = PATIENT,
    invite_code: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an account. Administrators call this directly; the UI uses register_account.

    When ``invite_code`` is given it is consumed in the same transaction as the
    insert, so a failed registration never burns a code and a code cannot be used twice.
    """
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
            if invite_code is not None:
                now = _iso(_now())
                claimed = conn.execute(
                    "UPDATE coordinator_invites SET used_at = ?, used_by = ? "
                    "WHERE code_hash = ? AND used_at IS NULL AND expires_at > ?",
                    (now, user_id, _digest(invite_code.strip()), now),
                ).rowcount
                if claimed != 1:
                    raise AuthError(INVALID_INVITE)
    except sqlite3.IntegrityError:
        raise AuthError(
            f"A {role} account with this email already exists."
        ) from None
    return {"id": user_id, "full_name": name, "email": address, "role": role}


def register_patient(full_name: str, email: str, password: str, confirm: str) -> Dict[str, Any]:
    """Public registration. The role is always patient; callers cannot choose it."""
    if password != confirm:
        raise AuthError("Passwords do not match.")
    return create_user(full_name, email, password, PATIENT)


def demo_mode() -> bool:
    """Local classroom demos only: CTAI_DEMO_MODE must be exactly 'true' (any case)."""
    return os.environ.get(DEMO_MODE_ENV, "").strip().lower() == "true"


def register_account(
    full_name: str, email: str, password: str, confirm: str,
    role: str = PATIENT, invite_code: str = "",
) -> Dict[str, Any]:
    """Public registration.

    Coordinator accounts need a valid administrator-issued invitation code. Only
    when the server runs with CTAI_DEMO_MODE=true is the fixed classroom code
    "0000" also accepted.
    """
    if role != COORDINATOR:
        return register_patient(full_name, email, password, confirm)
    code = (invite_code or "").strip()
    if not code:
        raise AuthError("Enter the invitation code.")
    if password != confirm:
        raise AuthError("Passwords do not match.")
    if demo_mode() and secrets.compare_digest(code, DEMO_CODE):
        return create_user(full_name, email, password, COORDINATOR)
    return create_user(full_name, email, password, COORDINATOR, invite_code=code)


def create_invitation(hours: int = INVITE_HOURS) -> str:
    """Administrator-only (CLI). Returns the code once; only its hash is stored."""
    code = secrets.token_urlsafe(12)
    now = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO coordinator_invites (code_hash, created_at, expires_at) VALUES (?, ?, ?)",
            (_digest(code), _iso(now), _iso(now + timedelta(hours=hours))),
        )
    return code


def get_user_by_email(email: str, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Find an account. When two accounts share the email, a role is required."""
    address = normalize_email(email)
    query, args = "SELECT * FROM users WHERE email = ?", [address]
    if role is not None:
        query, args = query + " AND role = ?", args + [role]
    with closing(_connect()) as conn:
        rows = conn.execute(query, args).fetchall()
    if len(rows) > 1:
        raise AuthError("More than one account uses that email; specify the role.")
    return _public(rows[0]) if rows else None


# ---------------------------------------------------------------------------
# Login, sessions and rate limiting
# ---------------------------------------------------------------------------

def _recent_failures(conn: sqlite3.Connection, email: str, role: str) -> int:
    since = _iso(_now() - timedelta(minutes=LOCKOUT_MINUTES))
    return conn.execute(
        "SELECT COUNT(*) FROM login_attempts WHERE email = ? AND role = ? AND attempted_at >= ?",
        (email, role, since),
    ).fetchone()[0]


def is_locked_out(email: str, role: str) -> bool:
    with closing(_connect()) as conn:
        return _recent_failures(conn, normalize_email(email), role) >= MAX_FAILED_LOGINS


def authenticate(email: str, password: str, role: str) -> Dict[str, Any]:
    """Verify email + password + selected role; raises AuthError with a generic message.

    The role is part of the account's identity: a patient login can never reach
    the coordinator account that shares its email, or the other way round. Wrong
    email, password or role all give the same message and count toward the same
    per-(email, role) lockout.
    """
    address = normalize_email(email)
    if not address or not password:
        raise AuthError("Enter your email and password.")
    if role not in ROLES:
        raise AuthError("Choose Patient or Coordinator.")
    with closing(_connect()) as conn, conn:
        if _recent_failures(conn, address, role) >= MAX_FAILED_LOGINS:
            raise AuthError(
                f"Too many failed attempts. Try again in {LOCKOUT_MINUTES} minutes."
            )
        row = conn.execute(
            "SELECT * FROM users WHERE email = ? AND role = ?", (address, role)
        ).fetchone()
        # Unknown accounts still pay for one hash check so timing does not reveal them.
        stored = row["password_hash"] if row else _DUMMY_HASH
        try:
            valid = _hasher.verify(stored, password)
        except (VerificationError, InvalidHashError):
            valid = False
        if not (valid and row):
            conn.execute(
                "INSERT INTO login_attempts (email, role, attempted_at) VALUES (?, ?, ?)",
                (address, role, _iso(_now())),
            )
            conn.commit()  # keep the failure on record even though we raise
            raise AuthError(GENERIC_LOGIN_ERROR)
        conn.execute(
            "DELETE FROM login_attempts WHERE email = ? AND role = ?", (address, role)
        )
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


def request_password_reset(email: str, role: str) -> str:
    """
    Email a single-use reset token when the (email, role) account exists.

    Always returns the same notice so the response does not reveal whether an
    account exists. Raises AuthError only when email delivery is not configured.
    """
    if not email_delivery_configured():
        raise AuthError(
            "Password recovery is unavailable because email delivery is not "
            "configured. Ask your administrator to reset your password."
        )
    address = normalize_email(email)
    user = get_user_by_email(address, role) if role in ROLES else None
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
            f"A password reset was requested for your Clinical Trial AI {user['role']} account.\n\n"
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
        conn.execute(
            "DELETE FROM login_attempts WHERE email = ? AND role = ?",
            (row["email"], row["role"]),
        )


def _admin_account(conn: sqlite3.Connection, email: str, role: Optional[str]) -> sqlite3.Row:
    """Find exactly one account for an administrator action; ambiguity is an error."""
    address = normalize_email(email)
    query, args = "SELECT * FROM users WHERE email = ?", [address]
    if role is not None:
        if role not in ROLES:
            raise AuthError("Unknown role.")
        query, args = query + " AND role = ?", args + [role]
    rows = conn.execute(query, args).fetchall()
    if not rows:
        raise AuthError("No matching account.")
    if len(rows) > 1:
        raise AuthError("More than one account uses that email; specify --role.")
    return rows[0]


def admin_set_password(email: str, new: str, role: Optional[str] = None) -> None:
    """Administrator reset (command line); ends that one account's sessions."""
    problem = validate_password(new, normalize_email(email))
    if problem:
        raise AuthError(problem)
    with closing(_connect()) as conn, conn:
        row = _admin_account(conn, email, role)
        conn.execute(
            "UPDATE users SET password_hash = ?, reset_token_hash = NULL, "
            "reset_expires_at = NULL WHERE id = ?",
            (_hasher.hash(new), row["id"]),
        )
        _revoke_user_sessions(conn, row["id"])
        conn.execute(
            "DELETE FROM login_attempts WHERE email = ? AND role = ?",
            (row["email"], row["role"]),
        )


def set_role(email: str, role: str, from_role: Optional[str] = None) -> None:
    """
    Administrator action (command line only): convert an existing account to ``role``.

    Never happens automatically. ``from_role`` is required when the email has two
    accounts, and the move is refused if the target role already exists for that email.
    """
    if role not in ROLES:
        raise AuthError("Unknown role.")
    try:
        with closing(_connect()) as conn, conn:
            row = _admin_account(conn, email, from_role)
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, row["id"]))
            conn.execute("DELETE FROM auth_sessions WHERE user_id = ?", (row["id"],))
    except sqlite3.IntegrityError:
        raise AuthError(f"A {role} account with this email already exists.") from None
