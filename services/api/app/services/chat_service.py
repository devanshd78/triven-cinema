import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHAT_DIR = PROJECT_ROOT / "storage" / "chats"
CHAT_DB = CHAT_DIR / "chats.sqlite3"
_DB_LOCK = Lock()
_INITIALIZED = False
_CHAT_ID_RE = re.compile(r"^chat-[A-Za-z0-9._:-]{8,120}$")


class ChatError(RuntimeError):
    pass


def _connect() -> sqlite3.Connection:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(CHAT_DB, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_chat_store() -> None:
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
                CREATE TABLE IF NOT EXISTS studio_chats (
                    id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    workspace_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(workspace_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_studio_chats_workspace_updated
                    ON studio_chats(workspace_id, updated_at DESC);
                """
            )
            connection.commit()
        _INITIALIZED = True


def _serialize(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "created_at": int(row["created_at"]),
        "updated_at": int(row["updated_at"]),
        "workspace": json.loads(row["workspace_json"]),
    }


def list_chats(workspace_id: str) -> list[dict]:
    initialize_chat_store()
    limit = max(1, int(settings.chat_history_limit))
    with _DB_LOCK, closing(_connect()) as connection:
        rows = connection.execute(
            """
            SELECT * FROM studio_chats
            WHERE workspace_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        ).fetchall()
    return [_serialize(row) for row in rows]


def save_chat(
    workspace_id: str,
    chat_id: str,
    *,
    title: str,
    workspace: dict[str, Any],
    created_at: int,
    updated_at: int,
) -> dict:
    initialize_chat_store()
    if not _CHAT_ID_RE.fullmatch(chat_id):
        raise ChatError("Invalid chat id.")
    clean_title = " ".join(title.split()).strip()[:120] or "New chat"
    try:
        workspace_json = json.dumps(workspace, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ChatError("Chat workspace is not JSON serializable.") from exc
    max_bytes = max(64_000, int(settings.chat_workspace_max_bytes))
    if len(workspace_json.encode("utf-8")) > max_bytes:
        raise ChatError("Chat state is too large to save. Remove large embedded data and retry.")

    created = max(0, int(created_at))
    updated = max(created, int(updated_at))
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO studio_chats(id, workspace_id, title, workspace_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, id) DO UPDATE SET
                title = excluded.title,
                workspace_json = excluded.workspace_json,
                    updated_at = excluded.updated_at
                WHERE excluded.updated_at >= studio_chats.updated_at
            """,
            (chat_id, workspace_id, clean_title, workspace_json, created, updated),
        )
        limit = max(1, int(settings.chat_history_limit))
        connection.execute(
            """
            DELETE FROM studio_chats
            WHERE workspace_id = ? AND id NOT IN (
                SELECT id FROM studio_chats
                WHERE workspace_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            )
            """,
            (workspace_id, workspace_id, limit),
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM studio_chats WHERE workspace_id = ? AND id = ?",
            (workspace_id, chat_id),
        ).fetchone()
    if not row:
        raise ChatError("Unable to save chat.")
    return _serialize(row)


def delete_chat(workspace_id: str, chat_id: str) -> bool:
    initialize_chat_store()
    with _DB_LOCK, closing(_connect()) as connection:
        cursor = connection.execute(
            "DELETE FROM studio_chats WHERE workspace_id = ? AND id = ?",
            (workspace_id, chat_id),
        )
        connection.commit()
        return bool(cursor.rowcount)
