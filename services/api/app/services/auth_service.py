import base64
import hashlib
import hmac
import json
import secrets
import smtplib
import sqlite3
import ssl
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from threading import Lock

from fastapi import Request, Response

from app.core.config import settings
from app.services.identity_service import COOKIE_NAME as WORKSPACE_COOKIE_NAME, sign_workspace_id


PROJECT_ROOT = Path(__file__).resolve().parents[4]
AUTH_DIR = PROJECT_ROOT / "storage" / "auth"
AUTH_DB = AUTH_DIR / "auth.sqlite3"
AUTH_COOKIE_NAME = "triven_auth"
_DB_LOCK = Lock()
_INITIALIZED = False


class AuthError(RuntimeError):
    pass


class AuthRateLimitError(AuthError):
    pass


class AuthDeliveryError(AuthError):
    pass


def demo_login_enabled() -> bool:
    return settings.demo_auth_show_otp and not settings.is_production


def validate_auth_configuration() -> None:
    if not settings.is_production:
        return
    if not settings.auth_enabled:
        raise AuthError("Production requires AUTH_ENABLED=true.")
    if settings.demo_auth_show_otp:
        raise AuthError("Production requires DEMO_AUTH_SHOW_OTP=false and SMTP email delivery.")
    if len(settings.triven_secret_key.strip()) < 32:
        raise AuthError("Production requires a TRIVEN_SECRET_KEY of at least 32 characters.")
    if not settings.smtp_host.strip() or not settings.smtp_from_email.strip():
        raise AuthDeliveryError("Configure SMTP_HOST and SMTP_FROM_EMAIL for production login.")
    if not (settings.smtp_use_tls or settings.smtp_use_ssl):
        raise AuthDeliveryError("Production SMTP requires TLS or SSL.")


def _deliver_otp(email: str, otp: str, ttl: int) -> None:
    if demo_login_enabled():
        return
    if not settings.smtp_host or not settings.smtp_from_email:
        raise AuthDeliveryError("Login email is not configured. Contact the administrator.")
    message = EmailMessage()
    message["Subject"] = "Your Triven Cinema sign-in code"
    message["From"] = settings.smtp_from_email
    message["To"] = email
    message.set_content(f"Your Triven Cinema sign-in code is {otp}.\n\nIt expires in {max(1, ttl // 60)} minutes. If you did not request it, ignore this email.")
    try:
        smtp_class = smtplib.SMTP_SSL if settings.smtp_use_ssl else smtplib.SMTP
        kwargs = {"timeout": settings.smtp_timeout_seconds}
        if settings.smtp_use_ssl:
            kwargs["context"] = ssl.create_default_context()
        with smtp_class(settings.smtp_host, settings.smtp_port, **kwargs) as smtp:
            if settings.smtp_use_tls and not settings.smtp_use_ssl:
                smtp.starttls(context=ssl.create_default_context())
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password)
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise AuthDeliveryError("The sign-in email could not be delivered. Try again later or contact the administrator.") from exc


def limit_otp_requests(client_key: str) -> None:
    initialize_auth_store()
    now = int(time.time())
    key = hmac.new(_secret(), client_key.encode(), hashlib.sha256).hexdigest()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute("DELETE FROM otp_request_limits WHERE window_start < ?", (now - 3600,))
        row = connection.execute("SELECT count FROM otp_request_limits WHERE key=?", (key,)).fetchone()
        if row and row["count"] >= max(1, settings.auth_otp_requests_per_hour):
            raise AuthRateLimitError("Too many sign-in requests. Please try again later.")
        connection.execute(
            "INSERT INTO otp_request_limits VALUES(?, ?, 1) ON CONFLICT(key) DO UPDATE SET count=count+1",
            (key, now),
        )
        connection.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secret() -> bytes:
    value = settings.triven_secret_key.strip()
    if value:
        return value.encode("utf-8")
    if settings.is_production:
        raise AuthError("TRIVEN_SECRET_KEY must be configured when Cinema login is enabled.")
    return b"triven-cinema-development-only"


def _connect() -> sqlite3.Connection:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(AUTH_DB, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_auth_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _DB_LOCK:
        if _INITIALIZED:
            return
        with closing(_connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    email_key TEXT NOT NULL UNIQUE,
                    workspace_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS otp_challenges (
                    id TEXT PRIMARY KEY,
                    email_key TEXT NOT NULL UNIQUE,
                    otp_digest TEXT NOT NULL,
                    expires_at INTEGER NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_users_workspace ON users(workspace_id);
                CREATE INDEX IF NOT EXISTS idx_otp_expires ON otp_challenges(expires_at);
                CREATE TABLE IF NOT EXISTS otp_request_limits (
                    key TEXT PRIMARY KEY, window_start INTEGER NOT NULL, count INTEGER NOT NULL
                );
                """
            )
            connection.commit()
        _INITIALIZED = True


def normalize_email(email: str) -> str:
    cleaned = email.strip().lower()
    if len(cleaned) > 254 or cleaned.count("@") != 1 or any(ch.isspace() or ord(ch) < 32 for ch in cleaned):
        raise AuthError("Enter a valid email address.")
    local, domain = cleaned.rsplit("@", 1)
    if not local or not domain or "." not in domain:
        raise AuthError("Enter a valid email address.")
    return cleaned


def _otp_digest(email_key: str, otp: str) -> str:
    return hmac.new(_secret(), f"otp:{email_key}:{otp}".encode("utf-8"), hashlib.sha256).hexdigest()


def request_otp(email: str) -> tuple[str, str, int]:
    validate_auth_configuration()
    initialize_auth_store()
    email_key = normalize_email(email)
    otp = f"{secrets.randbelow(1_000_000):06d}"
    challenge_id = uuid.uuid4().hex
    ttl = max(60, int(settings.auth_otp_ttl_seconds))
    expires_at = int(time.time()) + ttl
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute("DELETE FROM otp_challenges WHERE expires_at < ?", (int(time.time()),))
        existing = connection.execute("SELECT created_at FROM otp_challenges WHERE email_key=?", (email_key,)).fetchone()
        if existing and settings.is_production:
            elapsed = time.time() - datetime.fromisoformat(existing["created_at"]).timestamp()
            if elapsed < settings.auth_otp_resend_seconds:
                raise AuthRateLimitError("Please wait before requesting another sign-in code.")
        connection.execute("DELETE FROM otp_challenges WHERE email_key = ?", (email_key,))
        connection.execute(
            """
            INSERT INTO otp_challenges(id, email_key, otp_digest, expires_at, attempts, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (challenge_id, email_key, _otp_digest(email_key, otp), expires_at, _now()),
        )
        connection.commit()
    try:
        _deliver_otp(email_key, otp, ttl)
    except AuthDeliveryError:
        with _DB_LOCK, closing(_connect()) as connection:
            connection.execute("DELETE FROM otp_challenges WHERE id=?", (challenge_id,))
            connection.commit()
        raise
    return challenge_id, otp, ttl


def _valid_workspace_id(value: str | None) -> bool:
    return bool(value and len(value) == 32 and all(ch in "0123456789abcdef" for ch in value.lower()))


def _get_or_create_user(connection: sqlite3.Connection, email_key: str, preferred_workspace_id: str | None) -> dict:
    row = connection.execute("SELECT * FROM users WHERE email_key = ?", (email_key,)).fetchone()
    if row:
        connection.execute("UPDATE users SET updated_at = ? WHERE id = ?", (_now(), row["id"]))
        return dict(row)

    workspace_id = preferred_workspace_id.lower() if _valid_workspace_id(preferred_workspace_id) else uuid.uuid4().hex
    if connection.execute("SELECT 1 FROM users WHERE workspace_id = ?", (workspace_id,)).fetchone():
        workspace_id = uuid.uuid4().hex
    user_id = uuid.uuid4().hex
    timestamp = _now()
    connection.execute(
        """
        INSERT INTO users(id, email, email_key, workspace_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, email_key, email_key, workspace_id, timestamp, timestamp),
    )
    return {
        "id": user_id,
        "email": email_key,
        "email_key": email_key,
        "workspace_id": workspace_id,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def verify_otp(email: str, otp: str, *, preferred_workspace_id: str | None = None) -> dict:
    initialize_auth_store()
    email_key = normalize_email(email)
    cleaned_otp = "".join(ch for ch in otp if ch.isdigit())
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM otp_challenges WHERE email_key = ?", (email_key,)
        ).fetchone()
        if not row:
            raise AuthError("OTP expired or not requested. Request a new code.")
        if int(row["expires_at"]) < int(time.time()):
            connection.execute("DELETE FROM otp_challenges WHERE id = ?", (row["id"],))
            connection.commit()
            raise AuthError("OTP expired. Request a new code.")
        if int(row["attempts"]) >= max(1, int(settings.auth_otp_max_attempts)):
            connection.execute("DELETE FROM otp_challenges WHERE id = ?", (row["id"],))
            connection.commit()
            raise AuthError("Too many OTP attempts. Request a new code.")

        expected = str(row["otp_digest"])
        supplied = _otp_digest(email_key, cleaned_otp)
        if not hmac.compare_digest(expected, supplied):
            connection.execute(
                "UPDATE otp_challenges SET attempts = attempts + 1 WHERE id = ?", (row["id"],)
            )
            connection.commit()
            raise AuthError("Incorrect OTP.")

        connection.execute("DELETE FROM otp_challenges WHERE id = ?", (row["id"],))
        user = _get_or_create_user(connection, email_key, preferred_workspace_id)
        connection.commit()
        return user


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _session_signature(payload: str) -> str:
    digest = hmac.new(_secret(), f"auth:{payload}".encode("utf-8"), hashlib.sha256).digest()
    return _b64encode(digest)


def sign_auth_user(user: dict) -> str:
    max_age = max(3600, int(settings.auth_session_days) * 86400)
    payload = {
        "uid": str(user["id"]),
        "email": str(user["email"]),
        "workspace_id": str(user["workspace_id"]),
        "exp": int(time.time()) + max_age,
    }
    encoded = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{encoded}.{_session_signature(encoded)}"


def verify_auth_token(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    encoded, supplied = token.split(".", 1)
    try:
        expected = _session_signature(encoded)
        if not hmac.compare_digest(supplied, expected):
            return None
        payload = json.loads(_b64decode(encoded).decode("utf-8"))
        if int(payload.get("exp") or 0) < int(time.time()):
            return None
        user_id = str(payload.get("uid") or "")
        workspace_id = str(payload.get("workspace_id") or "")
        email = normalize_email(str(payload.get("email") or ""))
        if len(user_id) != 32 or not _valid_workspace_id(workspace_id):
            return None
    except (ValueError, TypeError, json.JSONDecodeError, AuthError):
        return None

    initialize_auth_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT id, email, workspace_id FROM users WHERE id = ? AND email_key = ? AND workspace_id = ?",
            (user_id, email, workspace_id),
        ).fetchone()
    return dict(row) if row else None


def auth_user_from_request(request: Request) -> dict | None:
    # Middleware and route dependencies often ask for the same authenticated
    # account during one request. Cache the verified row on request.state so
    # polling-heavy generation endpoints do not hit SQLite twice per request.
    if getattr(request.state, "_triven_auth_checked", False):
        return getattr(request.state, "triven_auth_user", None)
    user = verify_auth_token(request.cookies.get(AUTH_COOKIE_NAME))
    request.state.triven_auth_user = user
    request.state._triven_auth_checked = True
    return user


def authenticated_workspace_id(request: Request) -> str | None:
    user = auth_user_from_request(request)
    return str(user["workspace_id"]) if user else None


def set_login_cookies(response: Response, user: dict) -> None:
    max_age = max(3600, int(settings.auth_session_days) * 86400)
    response.set_cookie(
        AUTH_COOKIE_NAME,
        sign_auth_user(user),
        max_age=max_age,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        WORKSPACE_COOKIE_NAME,
        sign_workspace_id(str(user["workspace_id"])),
        max_age=max_age,
        httponly=True,
        secure=settings.is_production,
        samesite="lax",
        path="/",
    )


def clear_login_cookies(response: Response) -> None:
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    response.delete_cookie(WORKSPACE_COOKIE_NAME, path="/")
