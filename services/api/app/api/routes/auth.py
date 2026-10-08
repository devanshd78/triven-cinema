from fastapi import APIRouter, HTTPException, Request, Response

from app.core.config import settings
from app.schemas.auth import AuthMeResponse, AuthUserResponse, RequestOtpRequest, RequestOtpResponse, VerifyOtpRequest
from app.services.auth_service import (
    AuthError,
    AuthDeliveryError,
    AuthRateLimitError,
    auth_user_from_request,
    clear_login_cookies,
    demo_login_enabled,
    limit_otp_requests,
    request_otp,
    set_login_cookies,
    verify_otp,
)
from app.services.identity_service import workspace_id_from_request


router = APIRouter()


def _user_response(user: dict) -> AuthUserResponse:
    return AuthUserResponse(
        id=str(user["id"]),
        email=str(user["email"]),
        workspace_id=str(user["workspace_id"]),
    )


@router.post("/otp/request", response_model=RequestOtpResponse)
def request_login_otp(payload: RequestOtpRequest, request: Request) -> RequestOtpResponse:
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Cinema login is disabled.")
    try:
        limit_otp_requests(request.client.host if request.client else "unknown")
        challenge_id, otp, ttl = request_otp(payload.email)
    except AuthRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except AuthDeliveryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RequestOtpResponse(
        challenge_id=challenge_id,
        expires_in_seconds=ttl,
        demo_otp=otp if demo_login_enabled() else None,
        demo_mode=demo_login_enabled(),
    )


@router.post("/otp/verify", response_model=AuthMeResponse)
def verify_login_otp(payload: VerifyOtpRequest, request: Request, response: Response) -> AuthMeResponse:
    if not settings.auth_enabled:
        raise HTTPException(status_code=404, detail="Cinema login is disabled.")
    try:
        user = verify_otp(
            payload.email,
            payload.otp,
            preferred_workspace_id=workspace_id_from_request(request),
        )
        set_login_cookies(response, user)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AuthMeResponse(authenticated=True, user=_user_response(user))


@router.get("/me", response_model=AuthMeResponse)
def auth_me(request: Request) -> AuthMeResponse:
    if not settings.auth_enabled:
        return AuthMeResponse(authenticated=True, user=None)
    user = auth_user_from_request(request)
    if not user:
        return AuthMeResponse(authenticated=False, user=None)
    return AuthMeResponse(authenticated=True, user=_user_response(user))


@router.post("/logout")
def logout(response: Response) -> dict:
    clear_login_cookies(response)
    return {"ok": True}
