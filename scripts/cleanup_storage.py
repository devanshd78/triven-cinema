#!/usr/bin/env python3
import argparse
import os
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage"
GENERATED = STORAGE / "generated"
JOBS_DB = STORAGE / "jobs" / "jobs.sqlite3"
ENV_PATH = ROOT / ".env"


def env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is not None:
        return value
    if not ENV_PATH.exists():
        return None
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() == name:
            return candidate.strip().strip('"').strip("'")
    return None


def env_int(name: str, default: int) -> int:
    try:
        return max(1, int(env_value(name) or str(default)))
    except ValueError:
        return default


def expired(path: Path, days: int, now: float) -> bool:
    return now - path.stat().st_mtime > days * 86400


def cleanup_generated(apply: bool) -> tuple[int, int]:
    now = time.time()
    preview_days = env_int("PREVIEW_RETENTION_DAYS", 3)
    final_days = env_int("FINAL_RETENTION_DAYS", 30)
    removed = 0
    bytes_removed = 0
    if not GENERATED.exists():
        return removed, bytes_removed

    owned: set[str] = set()
    active: set[str] = set()
    if JOBS_DB.exists():
        with sqlite3.connect(JOBS_DB) as connection:
            has_assets = connection.execute("SELECT 1 FROM sqlite_master WHERE name='generated_assets'").fetchone()
            if has_assets:
                owned = {row[0] for row in connection.execute("SELECT filename FROM generated_assets")}
                active = {row[0] for row in connection.execute(
                    "SELECT a.filename FROM generated_assets a JOIN generation_jobs j ON j.id=a.job_id WHERE j.status IN ('queued','running')"
                )}

    for path in GENERATED.iterdir():
        if not path.is_file() or path.name.endswith(".part") or path.name in active:
            continue
        days = final_days if path.name in owned or path.name.startswith(("final-", "factory-")) else preview_days
        if not expired(path, days, now):
            continue
        size = path.stat().st_size
        print(f"{'DELETE' if apply else 'WOULD DELETE'} {path.name} ({size / (1024**2):.1f} MiB)")
        if apply:
            path.unlink(missing_ok=True)
        removed += 1
        bytes_removed += size
    return removed, bytes_removed


def prune_jobs(apply: bool) -> int:
    if not JOBS_DB.exists():
        return 0
    retention = max(env_int("JOB_RETENTION_DAYS", 14), env_int("FINAL_RETENTION_DAYS", 30))
    cutoff = time.time() - retention * 86400
    cutoff_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(cutoff))
    connection = sqlite3.connect(JOBS_DB)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM generation_jobs WHERE status IN ('completed','failed') AND updated_at < ?",
            (cutoff_iso,),
        ).fetchone()
        count = int(row[0] if row else 0)
        print(f"{'DELETE' if apply else 'WOULD DELETE'} {count} old completed/failed job record(s)")
        if apply and count:
            connection.execute(
                "DELETE FROM generation_jobs WHERE status IN ('completed','failed') AND updated_at < ?",
                (cutoff_iso,),
            )
            connection.commit()
        return count
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually delete expired data")
    args = parser.parse_args()
    GENERATED.mkdir(parents=True, exist_ok=True)
    files, bytes_removed = cleanup_generated(args.apply)
    jobs = prune_jobs(args.apply)
    print(
        f"Summary: files={files}, jobs={jobs}, reclaimable={bytes_removed/(1024**2):.1f} MiB, "
        f"mode={'apply' if args.apply else 'dry-run'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
