import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import auth, billing, chats, elements, factory, generations, health, youtube
from app.core.config import settings
from app.services.auth_service import auth_user_from_request, authenticated_workspace_id, initialize_auth_store, validate_auth_configuration
from app.services.billing_service import initialize_billing_store
from app.services.chat_service import initialize_chat_store
from app.services.element_service import initialize_element_store
from app.services.identity_service import (
    WORKSPACE_HEADER,
    ensure_workspace,
    workspace_id_from_request,
)
from app.services.job_service import (
    initialize_job_store,
    shutdown_job_executor,
    workspace_owns_generated_file,
)
from app.services.storage_service import resolve_generated_asset
from app.services.youtube_service import initialize_youtube_store


LOGGER = logging.getLogger("triven.api")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
STORAGE_DIR = PROJECT_ROOT / "storage"
GENERATED_DIR = STORAGE_DIR / "generated"
LOGS_DIR = STORAGE_DIR / "logs"

for directory in (STORAGE_DIR, GENERATED_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_auth_configuration()
    initialize_job_store()
    initialize_auth_store()
    initialize_chat_store()
    initialize_element_store()
    initialize_billing_store()
    initialize_youtube_store()
    LOGGER.info(
        "Triven Cinema API starting env=%s provider=%s",
        settings.app_env,
        settings.video_provider,
    )
    yield
    shutdown_job_executor()
    LOGGER.info("Triven Cinema API stopped")


app = FastAPI(
    title=settings.app_name,
    version="0.2.0",
    debug=settings.debug and not settings.is_production,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
    lifespan=lifespan,
)


# Production browser traffic should normally stay same-origin through Next.js.
# CORS is kept only for explicitly configured origins (useful during development).
if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID", WORKSPACE_HEADER],
        expose_headers=[WORKSPACE_HEADER],
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    started = time.perf_counter()

    if settings.auth_enabled and request.method != "OPTIONS" and request.url.path.startswith("/api/v1/"):
        public_prefixes = (
            "/api/v1/auth/",
            "/api/v1/health",
            "/api/v1/billing/webhook",
            "/api/v1/youtube/callback",
            "/api/v1/elements/assets/",
        )
        if not request.url.path.startswith(public_prefixes) and auth_user_from_request(request) is None:
            response = JSONResponse(status_code=401, content={"detail": "Authentication required."})
            response.headers["X-Request-ID"] = request_id
            response.headers["Cache-Control"] = "no-store"
            return response
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception("Unhandled request error request_id=%s path=%s", request_id, request.url.path)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    quiet_poll = (
        request.method == "GET"
        and (
            request.url.path.startswith("/api/v1/generations/jobs/")
            or request.url.path.startswith("/api/v1/health")
        )
    )
    if response.status_code >= 400 or not quiet_poll:
        LOGGER.info(
            "%s %s %s %.1fms request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
    return response


@app.post("/api/v1/identity/bootstrap")
def bootstrap_workspace(request: Request, response: Response) -> dict:
    """Establish one workspace identity before parallel UI bootstrap calls.

    The endpoint prevents several first-load billing/YouTube requests from each
    minting a different workspace concurrently. Production keeps the signed token
    HttpOnly; development may also receive a signed response header for split-origin
    localhost testing.
    """
    workspace_id = ensure_workspace(
        request,
        response,
        preferred_workspace_id=authenticated_workspace_id(request),
    )
    return {"status": "ready", "workspace_id": workspace_id}


# SECURITY: generated media is served through an ownership check rather than a
# raw StaticFiles mount. Assets are registered when produced, independently of
# whether later quality checks or publishing succeed.
@app.get("/media/generated/{filename}")
async def generated_media(filename: str, request: Request):
    try:
        path = resolve_generated_asset(filename, extensions={".mp4", ".png", ".jpg", ".jpeg", ".webp"})
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Generated media not found.") from exc

    if settings.auth_enabled or settings.is_production:
        workspace_id = workspace_id_from_request(request)
        if not workspace_id or not workspace_owns_generated_file(workspace_id, path.name):
            raise HTTPException(status_code=404, detail="Generated media not found.")

    media_type = "video/mp4" if path.suffix.lower() == ".mp4" else None
    return FileResponse(path, media_type=media_type, filename=None)

app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(chats.router, prefix="/api/v1/chats", tags=["Chats"])
app.include_router(generations.router, prefix="/api/v1/generations", tags=["Generations"])
app.include_router(factory.router, prefix="/api/v1/factory", tags=["Factory"])
app.include_router(elements.router, prefix="/api/v1/elements", tags=["Elements"])
app.include_router(billing.router, prefix="/api/v1/billing", tags=["Billing"])
app.include_router(youtube.router, prefix="/api/v1/youtube", tags=["YouTube"])


@app.get("/")
async def root():
    payload = {
        "name": settings.app_name,
        "status": "running",
        "environment": settings.app_env,
    }
    if not settings.is_production:
        payload["docs"] = "/docs"
    return payload
