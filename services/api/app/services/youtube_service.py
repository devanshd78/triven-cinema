import base64
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from urllib.parse import urlencode

import httpx
import requests
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings
from app.services.storage_service import resolve_generated_asset


PROJECT_ROOT = Path(__file__).resolve().parents[4]
INTEGRATIONS_DIR = PROJECT_ROOT / "storage" / "integrations"
INTEGRATIONS_DB = INTEGRATIONS_DIR / "integrations.sqlite3"
_DB_LOCK = Lock()
_INITIALIZED = False
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


class YouTubeIntegrationError(RuntimeError):
    pass


UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
UPLOAD_MAX_RETRIES = 5


def _resume_offset(range_header: str | None) -> int:
    """Return the next byte offset from a YouTube resumable-upload Range header."""
    if not range_header:
        return 0
    value = range_header.strip().lower()
    if not value.startswith("bytes=") or "-" not in value:
        return 0
    try:
        return int(value.rsplit("-", 1)[1]) + 1
    except ValueError:
        return 0


def _query_upload_offset(upload_url: str, total_size: int, access_token: str) -> int | None:
    try:
        response = requests.put(
            upload_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Length": "0",
                "Content-Range": f"bytes */{total_size}",
            },
            timeout=(20, 30),
        )
    except requests.RequestException:
        return None
    if response.status_code == 308:
        return _resume_offset(response.headers.get("Range"))
    if response.status_code in {200, 201}:
        return total_size
    return None


def _upload_resumable(
    upload_url: str,
    video_path: Path,
    access_token: str,
) -> dict:
    """Upload in resumable chunks so long 4K masters survive transient network faults."""
    total_size = video_path.stat().st_size
    if total_size <= 0:
        raise YouTubeIntegrationError("Generated video is empty.")

    offset = 0
    with video_path.open("rb") as handle:
        while offset < total_size:
            handle.seek(offset)
            chunk = handle.read(min(UPLOAD_CHUNK_BYTES, total_size - offset))
            if not chunk:
                raise YouTubeIntegrationError("Unexpected end of video during YouTube upload.")
            end = offset + len(chunk) - 1

            for attempt in range(UPLOAD_MAX_RETRIES):
                try:
                    response = requests.put(
                        upload_url,
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Content-Type": "video/mp4",
                            "Content-Length": str(len(chunk)),
                            "Content-Range": f"bytes {offset}-{end}/{total_size}",
                        },
                        data=chunk,
                        timeout=(30, 180),
                    )
                except requests.RequestException as exc:
                    if attempt + 1 >= UPLOAD_MAX_RETRIES:
                        raise YouTubeIntegrationError("YouTube video upload failed after retries.") from exc
                    time.sleep(min(8, 2**attempt))
                    resumed = _query_upload_offset(upload_url, total_size, access_token)
                    if resumed == total_size:
                        raise YouTubeIntegrationError(
                            "YouTube accepted the entire file but the final response was lost. "
                            "Check YouTube Studio before retrying to avoid a duplicate upload."
                        )
                    if resumed is not None and resumed > offset:
                        offset = resumed
                        break
                    continue

                if response.status_code in {200, 201}:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise YouTubeIntegrationError("YouTube returned an invalid upload response.") from exc

                if response.status_code == 308:
                    offset = _resume_offset(response.headers.get("Range"))
                    if offset <= 0:
                        offset = end + 1
                    break

                if response.status_code in {408, 429} or 500 <= response.status_code < 600:
                    if attempt + 1 >= UPLOAD_MAX_RETRIES:
                        raise YouTubeIntegrationError(
                            f"YouTube video upload failed after retries: {response.text[:400]}"
                        )
                    time.sleep(min(8, 2**attempt))
                    resumed = _query_upload_offset(upload_url, total_size, access_token)
                    if resumed == total_size:
                        raise YouTubeIntegrationError(
                            "YouTube accepted the entire file but the final response was lost. "
                            "Check YouTube Studio before retrying to avoid a duplicate upload."
                        )
                    if resumed is not None and resumed > offset:
                        offset = resumed
                        break
                    continue

                raise YouTubeIntegrationError(
                    f"YouTube video upload failed: {response.text[:400]}"
                )
            else:
                raise YouTubeIntegrationError("YouTube video upload failed after retries.")

    raise YouTubeIntegrationError("YouTube upload ended without a completed response.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    INTEGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(INTEGRATIONS_DB, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_youtube_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _DB_LOCK:
        if _INITIALIZED:
            return
        with closing(_connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS youtube_connections (
                    workspace_id TEXT PRIMARY KEY,
                    refresh_token_encrypted TEXT NOT NULL,
                    channel_id TEXT,
                    channel_title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS youtube_oauth_states (
                    state TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        _INITIALIZED = True


def _require_enabled() -> None:
    if not settings.youtube_enabled:
        raise YouTubeIntegrationError("YouTube integration is disabled.")
    if not settings.youtube_client_id.strip() or not settings.youtube_client_secret.strip():
        raise YouTubeIntegrationError("YouTube OAuth client credentials are not configured.")
    if not settings.triven_secret_key.strip():
        raise YouTubeIntegrationError("TRIVEN_SECRET_KEY is required for encrypted YouTube tokens.")


def _fernet() -> Fernet:
    secret = settings.triven_secret_key.strip()
    if not secret:
        raise YouTubeIntegrationError("TRIVEN_SECRET_KEY is not configured.")
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())
    return Fernet(key)


def _encrypt(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def _decrypt(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise YouTubeIntegrationError("Stored YouTube credentials cannot be decrypted.") from exc


def connection_status(workspace_id: str) -> dict:
    initialize_youtube_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT channel_id, channel_title FROM youtube_connections WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
    if not row:
        return {"connected": False, "channel_id": None, "channel_title": None}
    return {
        "connected": True,
        "channel_id": row["channel_id"],
        "channel_title": row["channel_title"],
    }


def create_authorization_url(workspace_id: str) -> str:
    _require_enabled()
    initialize_youtube_store()
    state = secrets.token_urlsafe(32)
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            "DELETE FROM youtube_oauth_states WHERE workspace_id = ?", (workspace_id,)
        )
        connection.execute(
            """
            INSERT INTO youtube_oauth_states(state, workspace_id, expires_at, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (state, workspace_id, expires_at, _now()),
        )
        connection.commit()

    query = urlencode(
        {
            "client_id": settings.youtube_client_id,
            "redirect_uri": settings.youtube_callback_url,
            "response_type": "code",
            "scope": " ".join(SCOPES),
            "access_type": "offline",
            "include_granted_scopes": "true",
            "prompt": "consent",
            "state": state,
        }
    )
    return f"https://accounts.google.com/o/oauth2/v2/auth?{query}"


def _consume_state(state: str) -> str:
    initialize_youtube_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT workspace_id, expires_at FROM youtube_oauth_states WHERE state = ?",
            (state,),
        ).fetchone()
        connection.execute("DELETE FROM youtube_oauth_states WHERE state = ?", (state,))
        connection.commit()
    if not row:
        raise YouTubeIntegrationError("YouTube OAuth state is invalid or already used.")
    expires = datetime.fromisoformat(str(row["expires_at"]))
    if expires < datetime.now(timezone.utc):
        raise YouTubeIntegrationError("YouTube OAuth state expired. Connect the channel again.")
    return str(row["workspace_id"])


def _token_exchange(code: str) -> dict:
    try:
        response = httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.youtube_client_id,
                "client_secret": settings.youtube_client_secret,
                "redirect_uri": settings.youtube_callback_url,
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise YouTubeIntegrationError("Could not exchange the YouTube authorization code.") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise YouTubeIntegrationError("Google returned an invalid OAuth response.") from exc
    if response.status_code >= 400:
        raise YouTubeIntegrationError(str(payload.get("error_description") or payload.get("error") or "YouTube OAuth failed."))
    return payload


def _refresh_access_token(refresh_token: str) -> str:
    try:
        response = httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.youtube_client_id,
                "client_secret": settings.youtube_client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise YouTubeIntegrationError("Unable to refresh YouTube access token.") from exc
    if response.status_code >= 400 or not payload.get("access_token"):
        raise YouTubeIntegrationError(str(payload.get("error_description") or payload.get("error") or "Unable to refresh YouTube access token."))
    return str(payload["access_token"])


def _channel(access_token: str) -> tuple[str | None, str | None]:
    try:
        response = httpx.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "id,snippet", "mine": "true"},
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise YouTubeIntegrationError("Unable to read the connected YouTube channel.") from exc
    if response.status_code >= 400:
        raise YouTubeIntegrationError("Google rejected the YouTube channel lookup.")
    items = payload.get("items") or []
    if not items:
        return None, None
    item = items[0]
    return str(item.get("id") or "") or None, str((item.get("snippet") or {}).get("title") or "") or None


def complete_oauth(state: str, code: str) -> tuple[str, str | None, str | None]:
    _require_enabled()
    workspace_id = _consume_state(state)
    tokens = _token_exchange(code)
    access_token = str(tokens.get("access_token") or "")
    refresh_token = str(tokens.get("refresh_token") or "")
    if not access_token:
        raise YouTubeIntegrationError("Google OAuth response did not include an access token.")

    if not refresh_token:
        with _DB_LOCK, closing(_connect()) as connection:
            row = connection.execute(
                "SELECT refresh_token_encrypted FROM youtube_connections WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        if row:
            refresh_token = _decrypt(str(row["refresh_token_encrypted"]))
    if not refresh_token:
        raise YouTubeIntegrationError(
            "Google did not return a refresh token. Reconnect with consent so offline uploads can work."
        )

    channel_id, channel_title = _channel(access_token)
    timestamp = _now()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO youtube_connections(
                workspace_id, refresh_token_encrypted, channel_id, channel_title, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
                refresh_token_encrypted = excluded.refresh_token_encrypted,
                channel_id = excluded.channel_id,
                channel_title = excluded.channel_title,
                updated_at = excluded.updated_at
            """,
            (workspace_id, _encrypt(refresh_token), channel_id, channel_title, timestamp, timestamp),
        )
        connection.commit()
    return workspace_id, channel_id, channel_title


def disconnect(workspace_id: str) -> None:
    initialize_youtube_store()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute("DELETE FROM youtube_connections WHERE workspace_id = ?", (workspace_id,))
        connection.commit()


def _refresh_token_for(workspace_id: str) -> str:
    initialize_youtube_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT refresh_token_encrypted FROM youtube_connections WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
    if not row:
        raise YouTubeIntegrationError("No YouTube channel is connected to this workspace.")
    return _decrypt(str(row["refresh_token_encrypted"]))


def upload_video(
    workspace_id: str,
    *,
    filename: str,
    title: str,
    description: str,
    privacy: str = "private",
    tags: list[str] | None = None,
    category_id: str = "22",
    publish_at: str | None = None,
) -> dict:
    _require_enabled()
    from app.services.job_service import workspace_owns_generated_file
    if not workspace_owns_generated_file(workspace_id, filename):
        raise YouTubeIntegrationError("Video not found in this account.")
    video_path = resolve_generated_asset(filename, extensions={".mp4"})
    refresh_token = _refresh_token_for(workspace_id)
    access_token = _refresh_access_token(refresh_token)

    effective_privacy = privacy
    if privacy in {"public", "unlisted"} and not settings.youtube_allow_public:
        effective_privacy = "private"

    status: dict[str, object] = {
        "privacyStatus": effective_privacy,
        "selfDeclaredMadeForKids": False,
    }
    if publish_at:
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at

    metadata = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": (tags or [])[:30],
            "categoryId": category_id or "22",
        },
        "status": status,
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(video_path.stat().st_size),
    }
    try:
        init = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos",
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers=headers,
            data=json.dumps(metadata),
            timeout=30,
        )
    except requests.RequestException as exc:
        raise YouTubeIntegrationError("Unable to start the YouTube resumable upload.") from exc
    if init.status_code >= 400:
        raise YouTubeIntegrationError(f"YouTube upload initialization failed: {init.text[:400]}")
    upload_url = init.headers.get("Location")
    if not upload_url:
        raise YouTubeIntegrationError("YouTube did not return a resumable upload URL.")

    payload = _upload_resumable(upload_url, video_path, access_token)
    video_id = str(payload.get("id") or "")
    if not video_id:
        raise YouTubeIntegrationError("YouTube upload completed without a video id.")
    return {
        "video_id": video_id,
        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
        "privacy": str(status["privacyStatus"]),
        "title": title[:100],
    }


initialize_youtube_store()
