import base64
import hashlib
import hmac
import time
import uuid

from fastapi import Request, Response

from app.core.config import settings


COOKIE_NAME = "triven_workspace"
WORKSPACE_HEADER = "X-Triven-Workspace"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365


class WorkspaceIdentityError(RuntimeError):
    pass


def _secret() -> bytes:
    value = settings.triven_secret_key.strip()
    if value:
        return value.encode("utf-8")
    if settings.is_production:
        raise WorkspaceIdentityError(
            "TRIVEN_SECRET_KEY must be configured in production."
        )
    return b"triven-cinema-development-only"


def _signature(workspace_id: str) -> str:
    digest = hmac.new(_secret(), workspace_id.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def sign_workspace_id(workspace_id: str) -> str:
    clean = workspace_id.strip().lower()
    if len(clean) != 32 or any(ch not in "0123456789abcdef" for ch in clean):
        raise WorkspaceIdentityError("Invalid workspace id.")
    return f"{clean}.{_signature(clean)}"


def verify_workspace_token(token: str | None) -> str | None:
    if not token or "." not in token:
        return None
    workspace_id, supplied = token.split(".", 1)
    try:
        expected = _signature(workspace_id)
    except WorkspaceIdentityError:
        raise
    if not hmac.compare_digest(supplied, expected):
        return None
    if len(workspace_id) != 32 or any(ch not in "0123456789abcdef" for ch in workspace_id):
        return None
    return workspace_id




def _asset_signature(workspace_id: str, asset_id: str, expires_at: int) -> str:
    payload = f"element-asset:{workspace_id}:{asset_id}:{expires_at}"
    digest = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def sign_element_asset_access(workspace_id: str, asset_id: str, *, ttl_seconds: int | None = None) -> str:
    ttl = int(ttl_seconds or settings.element_asset_url_ttl_seconds)
    expires_at = int(time.time()) + max(60, ttl)
    return f"{expires_at}.{_asset_signature(workspace_id, asset_id, expires_at)}"


def verify_element_asset_access(workspace_id: str, asset_id: str, token: str | None) -> bool:
    if not token or "." not in token:
        return False
    expires_raw, supplied = token.split(".", 1)
    try:
        expires_at = int(expires_raw)
    except ValueError:
        return False
    if expires_at < int(time.time()):
        return False
    expected = _asset_signature(workspace_id, asset_id, expires_at)
    return hmac.compare_digest(supplied, expected)


def workspace_id_from_request(request: Request) -> str | None:
    # The authenticated account is authoritative. An old signed workspace cookie
    # must never select a different user's Elements, media or billing records.
    if settings.auth_enabled:
        from app.services.auth_service import auth_user_from_request
        user = auth_user_from_request(request)
        if user:
            return str(user["workspace_id"])
        if settings.is_production:
            return None
    # Production identity remains HttpOnly-cookie based. During split-origin local
    # development (for example localhost:3000 -> 127.0.0.1:8000), browsers may
    # suppress SameSite cookies. A signed header fallback keeps the same identity
    # semantics without weakening production cookie isolation.
    cookie_workspace = verify_workspace_token(request.cookies.get(COOKIE_NAME))
    if cookie_workspace is not None:
        return cookie_workspace
    if not settings.is_production:
        return verify_workspace_token(request.headers.get(WORKSPACE_HEADER))
    return None


def ensure_workspace(
    request: Request,
    response: Response | None = None,
    *,
    preferred_workspace_id: str | None = None,
) -> str:
    workspace_id = workspace_id_from_request(request)
    if preferred_workspace_id is not None:
        clean_preferred = preferred_workspace_id.strip().lower()
        if len(clean_preferred) == 32 and all(ch in "0123456789abcdef" for ch in clean_preferred):
            workspace_id = clean_preferred
    if workspace_id is None:
        workspace_id = uuid.uuid4().hex

    if response is not None:
        token = sign_workspace_id(workspace_id)
        response.set_cookie(
            COOKIE_NAME,
            token,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            secure=settings.is_production,
            samesite="lax",
            path="/",
        )
        # Never expose the signed workspace token to browser JavaScript in
        # production. It is only a local-development fallback for split origins.
        if not settings.is_production:
            response.headers[WORKSPACE_HEADER] = token
    return workspace_id
