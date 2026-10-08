"""Idempotent request admission shared by generation entry points."""
import uuid
from threading import Lock
from typing import Callable

from fastapi import HTTPException

from app.core.config import settings
from app.schemas.generation import AsyncVideoGenerationResponse
from app.services.billing_service import BillingError, InsufficientCreditsError, consume_credits, refund_credits
from app.services.job_service import JobQueueFullError, find_job_by_request_id, get_job, submit_job


_ADMISSION_LOCK = Lock()


def submit_generation_request(job_type: str, payload, workspace_id: str,
                              runner: Callable[[str], dict], *, charge_seconds: int = 0) -> AsyncVideoGenerationResponse:
    # The supported deployment has one API process. Serialize the admission
    # transaction across billing and jobs so duplicates cannot refund each other.
    with _ADMISSION_LOCK:
        return _submit_generation_request(job_type, payload, workspace_id, runner, charge_seconds=charge_seconds)


def _submit_generation_request(job_type: str, payload, workspace_id: str,
                               runner: Callable[[str], dict], *, charge_seconds: int = 0) -> AsyncVideoGenerationResponse:
    data = payload.model_dump()
    data["workspace_id"] = workspace_id
    request_id = data.get("request_id")
    existing = find_job_by_request_id(workspace_id, request_id)
    if existing:
        if existing["job_type"] != job_type or any(existing["payload"].get(key) != value for key, value in data.items()):
            raise HTTPException(status_code=409, detail="This request ID belongs to another generation. Start a new take.")
        return AsyncVideoGenerationResponse(job_id=existing["job_id"], status=existing["status"],
                                            status_url=f"/api/v1/generations/jobs/{existing['job_id']}")
    reference = f"{job_type}:{workspace_id}:{request_id or uuid.uuid4().hex}"
    if charge_seconds:
        try:
            consume_credits(workspace_id, charge_seconds, reference)
        except InsufficientCreditsError as exc:
            raise HTTPException(status_code=402, detail=str(exc)) from exc
        except BillingError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    data.update(charge_reference=reference, charge_seconds=charge_seconds,
                credits_charged=bool(charge_seconds and settings.billing_enforce_credits))

    def charged_runner(job_id: str) -> dict:
        try:
            return runner(job_id)
        except Exception:
            if charge_seconds:
                refund_credits(workspace_id, charge_seconds, reference)
            raise

    try:
        job_id = submit_job(job_type, data, charged_runner)
    except Exception as exc:
        if charge_seconds and not find_job_by_request_id(workspace_id, request_id):
            refund_credits(workspace_id, charge_seconds, reference)
        if isinstance(exc, JobQueueFullError):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise
    current = get_job(job_id)
    return AsyncVideoGenerationResponse(job_id=job_id, status=(current or {}).get("status", "queued"),
                                        status_url=f"/api/v1/generations/jobs/{job_id}")
