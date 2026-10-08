import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.services.auth_service import AuthError, validate_auth_configuration


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parents[5]
STORAGE_DIR = PROJECT_ROOT / "storage"


def _storage_writable() -> bool:
    try:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        probe = STORAGE_DIR / ".health-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _disk_free_gb() -> float:
    usage = shutil.disk_usage(STORAGE_DIR)
    return round(usage.free / (1024**3), 2)


@router.get("")
async def health_check():
    return {
        "status": "ok",
        "service": "triven-cinema-api",
        "environment": settings.app_env,
    }


@router.get("/ready")
async def readiness_check():
    try:
        validate_auth_configuration()
        auth_ready = True
    except AuthError:
        auth_ready = False
    free_gb = _disk_free_gb()
    checks = {
        "storage_writable": _storage_writable(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "disk_free_gb": free_gb,
        "disk_ok": free_gb >= settings.minimum_free_disk_gb,
        "provider": settings.video_provider,
        "gemini_configured": bool(settings.gemini_api_key),
        "modal_config_present": bool(settings.modal_app_name and settings.modal_function_name),
        "authentication_configured": auth_ready,
        "inference_check": "Run scripts/check_inference.py; provider preflight also runs before every GPU request.",
    }
    ready = bool(
        checks["storage_writable"]
        and checks["ffmpeg"]
        and checks["ffprobe"]
        and checks["disk_ok"]
        and checks["authentication_configured"]
    )
    payload = {
        "status": "ready" if ready else "not_ready",
        "service": "triven-cinema-api",
        "checks": checks,
    }
    if not ready:
        raise HTTPException(status_code=503, detail=payload)
    return payload
