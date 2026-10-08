import json
import re
import sqlite3
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Callable

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[4]
JOBS_DIR = PROJECT_ROOT / "storage" / "jobs"
JOBS_DB = JOBS_DIR / "jobs.sqlite3"
_DB_LOCK = Lock()
_INITIALIZED = False
_EXECUTOR: ThreadPoolExecutor | None = None
_CURRENT_JOB: ContextVar[str | None] = ContextVar("triven_current_job", default=None)


class JobQueueFullError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(JOBS_DB, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        workers = max(1, int(settings.job_workers))
        _EXECUTOR = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="triven-job",
        )
    return _EXECUTOR


def _output_filenames(value: object) -> set[str]:
    """Only output fields confer ownership; prompts/reference input names never do."""
    filenames: set[str] = set()
    output_keys = {"filename", "final_filename", "continuity_frame_filename", "video_url",
                   "final_video_url", "continuity_frame_url", "download_url", "final_download_url"}
    if isinstance(value, dict):
        for key, item in value.items():
            if key in output_keys and isinstance(item, str):
                name = Path(item.split("?", 1)[0]).name
                if Path(name).suffix.lower() in {".mp4", ".png", ".jpg", ".jpeg", ".webp"}:
                    filenames.add(name)
            elif isinstance(item, (dict, list)):
                filenames.update(_output_filenames(item))
    elif isinstance(value, list):
        for item in value:
            filenames.update(_output_filenames(item))
    return filenames


def initialize_job_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    with _DB_LOCK:
        if _INITIALIZED:
            return
        with closing(_connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generation_jobs_updated_at ON generation_jobs(updated_at)"
            )
            # Keep ownership independently of transient job retention. Migrate old
            # completed results once, including jobs older than the former 500-row cap.
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(generation_jobs)")}
            for name in ("workspace_id", "request_id", "chat_id"):
                if name not in columns:
                    connection.execute(f"ALTER TABLE generation_jobs ADD COLUMN {name} TEXT")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS generated_assets (
                    filename TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    job_id TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assets_workspace ON generated_assets(workspace_id);
                CREATE INDEX IF NOT EXISTS idx_assets_job ON generated_assets(job_id);
                CREATE INDEX IF NOT EXISTS idx_jobs_workspace ON generation_jobs(workspace_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_jobs_chat ON generation_jobs(workspace_id, chat_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_request
                    ON generation_jobs(workspace_id, request_id) WHERE request_id IS NOT NULL;
            """)
            if connection.execute("PRAGMA user_version").fetchone()[0] < 2:
                for row in connection.execute("SELECT * FROM generation_jobs").fetchall():
                    payload = json.loads(row["payload_json"])
                    owner = str(payload.get("workspace_id") or "")
                    connection.execute(
                        "UPDATE generation_jobs SET workspace_id=?, chat_id=? WHERE id=?",
                        (owner, payload.get("chat_id"), row["id"]),
                    )
                    if owner and row["result_json"]:
                        for filename in _output_filenames(json.loads(row["result_json"])):
                            connection.execute(
                                "INSERT OR IGNORE INTO generated_assets VALUES(?,?,?,?,?,?)",
                                (filename, owner, row["id"], "{}", row["created_at"], row["updated_at"]),
                            )
                connection.execute("PRAGMA user_version=2")
            connection.execute(
                """
                UPDATE generation_jobs SET status='failed', stage='failed', progress=100,
                    error='The application restarted. Saved clips are available below; retry the unfinished work.',
                    message='Interrupted by application restart.', updated_at=?
                WHERE status IN ('queued','running')
                """, (_now(),)
            )
            interrupted = connection.execute(
                "SELECT id,payload_json FROM generation_jobs WHERE status='failed' AND message='Interrupted by application restart.'"
            ).fetchall()
            known_references = {
                data.get("charge_reference")
                for row in connection.execute("SELECT payload_json FROM generation_jobs")
                if (data := json.loads(row["payload_json"])).get("charge_reference")
            }
            completed_charges = [json.loads(row["payload_json"]) for row in connection.execute(
                "SELECT payload_json FROM generation_jobs WHERE status='completed'"
            )]
            connection.commit()
        _INITIALIZED = True
    from app.services.billing_service import reconcile_orphaned_generation_charges, settle_credits
    for payload in completed_charges:
        if payload.get("credits_charged") and payload.get("charge_reference"):
            settle_credits(str(payload["workspace_id"]), payload["charge_reference"])
    reconcile_orphaned_generation_charges(known_references)
    # A process crash cannot execute the runner's refund handler. The ledger's
    # unique reference makes reconciliation safe to repeat after another restart.
    for row in interrupted:
        payload = json.loads(row["payload_json"])
        if payload.get("credits_charged") and payload.get("charge_reference"):
            from app.services.billing_service import refund_credits
            refund_credits(str(payload["workspace_id"]), int(payload["charge_seconds"]), payload["charge_reference"])


def active_job_count() -> int:
    initialize_job_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM generation_jobs WHERE status IN ('queued', 'running')"
        ).fetchone()
    return int(row["count"] if row else 0)


def prune_old_jobs(retention_days: int | None = None) -> int:
    initialize_job_store()
    days = max(1, int(retention_days or settings.job_retention_days), int(settings.final_retention_days))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _DB_LOCK, closing(_connect()) as connection:
        cursor = connection.execute(
            """
            DELETE FROM generation_jobs
            WHERE status IN ('completed', 'failed') AND updated_at < ?
            """,
            (cutoff,),
        )
        connection.commit()
        return int(cursor.rowcount or 0)


def _create_job(job_type: str, payload: dict) -> tuple[str, bool]:
    initialize_job_store()
    max_pending = max(1, int(settings.job_max_pending))
    job_id = uuid.uuid4().hex
    timestamp = _now()
    owner = str(payload.get("workspace_id") or "")
    request_id = payload.get("request_id") or None
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if request_id:
            existing = connection.execute(
                "SELECT id, job_type, payload_json FROM generation_jobs WHERE workspace_id=? AND request_id=?",
                (owner, request_id),
            ).fetchone()
            if existing:
                old = json.loads(existing["payload_json"])
                ignored = {"charge_reference", "charge_seconds", "credits_charged"}
                if existing["job_type"] != job_type or any(old.get(k) != v for k, v in payload.items() if k not in ignored):
                    raise ValueError("This request ID already belongs to a different generation. Start a new take.")
                return str(existing["id"]), False
        active = connection.execute(
            "SELECT COUNT(*) FROM generation_jobs WHERE status IN ('queued','running')"
        ).fetchone()[0]
        if active >= max_pending:
            raise JobQueueFullError(
                f"Generation queue is full ({max_pending} active/pending jobs). Wait for a render to finish."
            )
        connection.execute(
            """INSERT INTO generation_jobs (
                id,job_type,status,stage,progress,message,payload_json,result_json,error,created_at,updated_at,
                workspace_id,request_id,chat_id
            ) VALUES (?,?,'queued','queued',0,'Queued',?,NULL,NULL,?,?,?,?,?)""",
            (job_id, job_type, json.dumps(payload), timestamp, timestamp, owner, request_id, payload.get("chat_id")),
        )
        connection.commit()
    return job_id, True


def create_job(job_type: str, payload: dict) -> str:
    return _create_job(job_type, payload)[0]


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    stage: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    initialize_job_store()
    if result is not None:
        with _DB_LOCK, closing(_connect()) as connection:
            row = connection.execute("SELECT workspace_id FROM generation_jobs WHERE id=?", (job_id,)).fetchone()
        if row and row["workspace_id"]:
            for filename in _output_filenames(result):
                register_generated_asset(row["workspace_id"], filename, job_id=job_id)
    fields: list[str] = ["updated_at = ?"]
    values: list[object] = [_now()]

    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if progress is not None:
        fields.append("progress = ?")
        values.append(max(0, min(100, int(progress))))
    if message is not None:
        fields.append("message = ?")
        values.append(message)
    if result is not None:
        fields.append("result_json = ?")
        values.append(json.dumps(result))
    if error is not None:
        fields.append("error = ?")
        values.append(error)

    values.append(job_id)
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            f"UPDATE generation_jobs SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        connection.commit()


def _asset_dict(row: sqlite3.Row) -> dict:
    return {
        "filename": row["filename"], "job_id": row["job_id"],
        "video_url": f"/media/generated/{row['filename']}",
        "download_url": f"/api/v1/generations/download/{row['filename']}",
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def register_generated_asset(workspace_id: str, filename: str, *, job_id: str | None = None,
                             metadata: dict | None = None) -> dict:
    initialize_job_store()
    if not workspace_id or Path(filename).name != filename or not filename:
        raise ValueError("An owned generated asset requires a workspace and a valid filename.")
    job_id = job_id or _CURRENT_JOB.get()
    timestamp = _now()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT * FROM generated_assets WHERE filename=?", (filename,)).fetchone()
        if row and row["workspace_id"] != workspace_id:
            raise ValueError("Generated asset belongs to another account.")
        merged = json.loads(row["metadata_json"]) if row else {}
        merged.update(metadata or {})
        # Keep the producing job stable if another job later references this asset.
        producer = row["job_id"] if row and row["job_id"] else job_id
        connection.execute(
            """INSERT INTO generated_assets VALUES(?,?,?,?,?,?)
            ON CONFLICT(filename) DO UPDATE SET job_id=excluded.job_id,
                metadata_json=excluded.metadata_json, updated_at=excluded.updated_at""",
            (filename, workspace_id, producer, json.dumps(merged), timestamp, timestamp),
        )
        connection.commit()
        return _asset_dict(connection.execute("SELECT * FROM generated_assets WHERE filename=?", (filename,)).fetchone())


def checkpoint_job_result(result: dict) -> None:
    job_id = _CURRENT_JOB.get()
    if not job_id:
        return
    job = get_job(job_id)
    merged = dict((job or {}).get("result") or {})
    merged.update(result)
    update_job(job_id, result=merged)


def register_current_generated_asset(filename: str, *, metadata: dict | None = None) -> dict | None:
    job_id = _CURRENT_JOB.get()
    if not job_id:
        return None
    job = get_job(job_id)
    owner = str(((job or {}).get("payload") or {}).get("workspace_id") or "")
    return register_generated_asset(owner, filename, job_id=job_id, metadata=metadata) if owner else None


def _job_dict(row: sqlite3.Row, assets: list[dict]) -> dict:
    return {
        "job_id": row["id"], "job_type": row["job_type"], "status": row["status"],
        "stage": row["stage"], "progress": row["progress"], "message": row["message"],
        "payload": json.loads(row["payload_json"]),
        "chat_id": row["chat_id"], "request_id": row["request_id"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "assets": assets, "error": row["error"], "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def get_job(job_id: str) -> dict | None:
    initialize_job_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute("SELECT * FROM generation_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        assets = [_asset_dict(asset) for asset in connection.execute(
            "SELECT * FROM generated_assets WHERE job_id=? AND workspace_id=? ORDER BY created_at",
            (job_id, row["workspace_id"]),
        )]
    return _job_dict(row, assets)


def list_jobs(workspace_id: str, *, chat_id: str | None = None, limit: int = 100) -> list[dict]:
    initialize_job_store()
    with _DB_LOCK, closing(_connect()) as connection:
        query = "SELECT * FROM generation_jobs WHERE workspace_id=?"
        params: list[object] = [workspace_id]
        if chat_id:
            query += " AND chat_id=?"
            params.append(chat_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, min(500, limit)))
        rows = connection.execute(query, params).fetchall()
        assets = connection.execute("SELECT * FROM generated_assets WHERE workspace_id=?", (workspace_id,)).fetchall()
    by_job: dict[str, list[dict]] = {}
    for asset in assets:
        by_job.setdefault(asset["job_id"], []).append(_asset_dict(asset))
    return [_job_dict(row, by_job.get(row["id"], [])) for row in rows]


def find_job_by_request_id(workspace_id: str, request_id: str | None) -> dict | None:
    if not request_id:
        return None
    initialize_job_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT id FROM generation_jobs WHERE workspace_id=? AND request_id=?",
            (workspace_id, request_id),
        ).fetchone()
    return get_job(row["id"]) if row else None


def workspace_owns_generated_file(workspace_id: str, filename: str) -> bool:
    if not workspace_id or not filename or Path(filename).name != filename:
        return False
    initialize_job_store()
    with _DB_LOCK, closing(_connect()) as connection:
        return connection.execute(
            "SELECT 1 FROM generated_assets WHERE filename=? AND workspace_id=?", (filename, workspace_id)
        ).fetchone() is not None


def generated_asset_metadata(workspace_id: str, filename: str) -> dict:
    initialize_job_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT metadata_json FROM generated_assets WHERE filename=? AND workspace_id=?", (filename, workspace_id)
        ).fetchone()
    if not row:
        raise ValueError("Generated video not found in this account.")
    return json.loads(row["metadata_json"])


def safe_job_error(exc: Exception, stage: str = "rendering") -> str:
    """Useful error categories without exposing provider credentials or stack traces."""
    detail = str(exc)
    lower = detail.lower()
    if isinstance(exc, ValueError):
        # Application validation errors are deliberately written for the caller.
        message = detail
    elif any(word in lower for word in ("weights", "checkpoint", "lora", "model file", "preflight", "worker version")):
        message = "Inference setup is incomplete or incompatible. Validate model weights, LoRAs and the deployed Modal worker before retrying."
    elif any(word in lower for word in ("unauthenticated", "authentication", "unauthorized", "token", "credential")):
        message = "The inference provider could not authenticate. Check the configured provider credentials before retrying."
    elif "429" in lower or "rate limit" in lower:
        message = "The provider rate limit was reached. Saved clips are retained; retry the unfinished step later."
    elif "timeout" in lower or "timed out" in lower:
        message = "The provider timed out. Saved clips are retained; check the job before starting another take."
    elif "disk" in lower or "space left" in lower:
        message = "Storage is low or unavailable. Free disk space before retrying the unfinished step."
    elif "out of memory" in lower or "cuda" in lower:
        message = "The GPU could not complete this render. Try a shorter scene or lower resolution."
    else:
        message = f"The {stage.replace('_', ' ')} step failed. Saved clips remain available; retry the unfinished step."
    message = re.sub(r"(?i)(bearer\s+|(?:token|secret|api[_-]?key)\s*[=:]\s*)\S+", r"\1[redacted]", message)
    return message[:600]


def submit_job(
    job_type: str,
    payload: dict,
    runner: Callable[[str], dict],
) -> str:
    job_id, created = _create_job(job_type, payload)
    if not created:
        return job_id

    def _run() -> None:
        context_token = _CURRENT_JOB.set(job_id)
        try:
            update_job(
                job_id,
                status="running",
                stage="initializing",
                progress=5,
                message="Initializing generation job...",
            )
            result = runner(job_id)
            update_job(
                job_id,
                status="completed",
                stage="completed",
                progress=100,
                message="Generation complete.",
                result=result,
            )
            if payload.get("credits_charged") and payload.get("charge_reference"):
                from app.services.billing_service import settle_credits
                # Result is already durable. A settlement error can be reconciled
                # on startup and must never turn delivered media into a failure.
                try:
                    settle_credits(str(payload["workspace_id"]), payload["charge_reference"])
                except Exception:
                    traceback.print_exc()
        except Exception as exc:  # noqa: BLE001 - job boundary must capture all failures
            traceback.print_exc()
            job = get_job(job_id)
            safe_error = safe_job_error(exc, (job or {}).get("stage") or "rendering")
            update_job(
                job_id,
                status="failed",
                stage="failed",
                progress=100,
                message="Generation failed.",
                error=safe_error,
            )
        finally:
            _CURRENT_JOB.reset(context_token)

    try:
        _executor().submit(_run)
    except Exception:
        update_job(job_id, status="failed", stage="failed", progress=100,
                   message="Unable to schedule generation.",
                   error="The generation worker could not start. Your credits were refunded; start a new take.")
        if payload.get("credits_charged") and payload.get("charge_reference"):
            from app.services.billing_service import refund_credits
            refund_credits(str(payload["workspace_id"]), int(payload["charge_seconds"]), payload["charge_reference"])
        raise
    return job_id


def shutdown_job_executor() -> None:
    global _EXECUTOR
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _EXECUTOR = None


initialize_job_store()
