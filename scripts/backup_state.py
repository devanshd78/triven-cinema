#!/usr/bin/env python3
import os
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage"
JOBS_DB = STORAGE / "jobs" / "jobs.sqlite3"
BILLING_DB = STORAGE / "billing" / "billing.sqlite3"
INTEGRATIONS_DB = STORAGE / "integrations" / "integrations.sqlite3"
AUTH_DB = STORAGE / "auth" / "auth.sqlite3"
CHATS_DB = STORAGE / "chats" / "chats.sqlite3"
ELEMENTS_DB = STORAGE / "elements" / "elements.sqlite3"
METRICS = STORAGE / "metrics"
BACKUPS = STORAGE / "backups"
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


def main() -> int:
    retention_days = env_int("BACKUP_RETENTION_DAYS", 7)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = BACKUPS / timestamp
    target.mkdir(parents=True, exist_ok=False)

    backed_up = 0
    databases = [
        (JOBS_DB, "jobs.sqlite3"),
        (BILLING_DB, "billing.sqlite3"),
        (INTEGRATIONS_DB, "integrations.sqlite3"),
        (AUTH_DB, "auth.sqlite3"),
        (CHATS_DB, "chats.sqlite3"),
        (ELEMENTS_DB, "elements.sqlite3"),
    ]
    for source, name in databases:
        if not source.exists():
            continue
        destination = target / name
        source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        dest_conn = sqlite3.connect(destination)
        try:
            source_conn.backup(dest_conn)
        finally:
            dest_conn.close()
            source_conn.close()
        backed_up += 1

    if METRICS.exists():
        target_metrics = target / "metrics"
        metric_files = [path for path in METRICS.glob("generations.jsonl*") if path.is_file()]
        if metric_files:
            target_metrics.mkdir(exist_ok=True)
            for path in metric_files:
                shutil.copy2(path, target_metrics / path.name)
                backed_up += 1

    # Element metadata cannot be restored without its uploaded reference images.
    element_assets = STORAGE / "elements" / "assets"
    if element_assets.exists():
        shutil.copytree(element_assets, target / "element-assets")
        backed_up += 1

    # Avoid accumulating empty backup directories on a fresh server.
    if backed_up == 0:
        target.rmdir()
        print("No SQLite/metrics state exists yet; backup skipped.")
    else:
        print(f"State backup created: {target}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    BACKUPS.mkdir(parents=True, exist_ok=True)
    pruned = 0
    for path in BACKUPS.iterdir():
        if not path.is_dir() or path == target:
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if modified < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            pruned += 1

    if pruned:
        print(f"Pruned {pruned} backup director{'y' if pruned == 1 else 'ies'} older than {retention_days} day(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
