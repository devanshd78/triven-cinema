import logging
import uuid

from fastapi import APIRouter, HTTPException, Request, Response

from app.schemas.factory import FactoryGenerationRequest
from app.schemas.generation import AsyncVideoGenerationResponse
from app.services.billing_service import (
    InsufficientCreditsError,
    consume_credits,
    refund_credits,
)
from app.services.factory_service import run_factory_generation
from app.services.job_request_service import submit_generation_request
from app.services.identity_service import ensure_workspace
from app.services.job_service import JobQueueFullError, submit_job, update_job


router = APIRouter()
LOGGER = logging.getLogger("triven.factory")


@router.post("/jobs", response_model=AsyncVideoGenerationResponse)
def create_factory_job(
    payload: FactoryGenerationRequest,
    request: Request,
    response: Response,
) -> AsyncVideoGenerationResponse:
    workspace_id = ensure_workspace(request, response)
    def runner(job_id: str) -> dict:
        def progress(stage: str, percent: int, message: str) -> None:
            update_job(job_id, status="running", stage=stage, progress=percent, message=message)
        return run_factory_generation(payload, workspace_id=workspace_id, progress=progress).model_dump()
    return submit_generation_request("factory", payload, workspace_id, runner,
                                     charge_seconds=max(1, int(round(payload.target_duration_seconds))))
