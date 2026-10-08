import json
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.elements import ElementListResponse, ElementResponse, ElementUpdateRequest
from app.services.element_service import (
    ElementError,
    UploadedElementAsset,
    add_element_assets,
    archive_element,
    create_element,
    get_element,
    list_elements,
    resolve_element_asset,
    resolve_element_asset_with_access,
    update_element,
)
from app.services.identity_service import ensure_workspace, workspace_id_from_request


router = APIRouter()


def _parse_roles(raw: str, count: int) -> list[str]:
    if not raw.strip():
        return ["support"] * count
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ElementError("Reference roles must be a JSON array.") from exc
    if not isinstance(value, list) or len(value) != count:
        raise ElementError("Reference roles must match the number of uploaded images.")
    allowed = {"primary", "face", "full_body", "profile", "costume", "object", "location", "style", "support"}
    roles = [str(item) for item in value]
    if any(role not in allowed for role in roles):
        raise ElementError("Unsupported Element reference role.")
    return roles


def _read_uploads(files: list[UploadFile], *, roles: str = "") -> list[UploadedElementAsset]:
    if len(files) > settings.element_max_assets_per_element:
        raise ElementError(f"Choose at most {settings.element_max_assets_per_element} reference images.")
    parsed_roles = _parse_roles(roles, len(files))
    uploads: list[UploadedElementAsset] = []
    max_bytes = max(1, int(settings.element_max_upload_mb)) * 1024 * 1024
    for index, file in enumerate(files):
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ElementError(f"{file.filename or 'Reference image'}: exceeds the {settings.element_max_upload_mb} MB upload limit.")
        uploads.append(
            UploadedElementAsset(
                original_filename=file.filename or "reference",
                content_type=file.content_type or "application/octet-stream",
                data=data,
                role=parsed_roles[index],
            )
        )
    return uploads


@router.get("", response_model=ElementListResponse)
def list_workspace_elements(request: Request, response: Response, include_archived: bool = False) -> ElementListResponse:
    workspace_id = ensure_workspace(request, response)
    elements = list_elements(workspace_id, include_archived=include_archived)
    return ElementListResponse(
        elements=elements,
        count=len(elements),
        max_stored=settings.element_max_stored_per_workspace,
        max_assets_per_element=settings.element_max_assets_per_element,
        max_active_per_scene=settings.element_max_active_per_scene,
    )


@router.post("", response_model=ElementResponse)
def create_workspace_element(
    request: Request,
    response: Response,
    name: str = Form(...),
    handle: str = Form(...),
    type: str = Form(...),
    description: str = Form(""),
    roles: str = Form(""),
    request_id: str = Form(""),
    files: list[UploadFile] = File(...),
) -> ElementResponse:
    workspace_id = ensure_workspace(request, response)
    try:
        return ElementResponse.model_validate(
            create_element(
                workspace_id,
                name=name,
                handle=handle,
                element_type=type,
                description=description,
                uploads=_read_uploads(files, roles=roles),
                request_id=request_id or None,
            )
        )
    except ElementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/assets/{asset_id}")
def read_element_asset(asset_id: str, request: Request, access: str | None = None):
    # Normal same-origin production requests are authorized by the signed
    # workspace cookie. Browser image tags cannot send X-Triven-Workspace, so a
    # short-lived asset-scoped signature is accepted as a safe fallback for
    # split-origin development and direct asset rendering.
    workspace_id = workspace_id_from_request(request)
    try:
        if workspace_id is not None:
            try:
                path, mime = resolve_element_asset(workspace_id, asset_id)
            except ElementError:
                path, mime = resolve_element_asset_with_access(asset_id, access)
        else:
            path, mime = resolve_element_asset_with_access(asset_id, access)
    except ElementError as exc:
        raise HTTPException(status_code=404, detail="Element asset not found.") from exc
    return FileResponse(
        path,
        media_type=mime,
        headers={
            "Cache-Control": "private, max-age=300",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/{element_id}", response_model=ElementResponse)
def read_workspace_element(element_id: str, request: Request, response: Response) -> ElementResponse:
    workspace_id = ensure_workspace(request, response)
    try:
        return ElementResponse.model_validate(get_element(workspace_id, element_id))
    except ElementError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{element_id}", response_model=ElementResponse)
def patch_workspace_element(
    element_id: str,
    payload: ElementUpdateRequest,
    request: Request,
    response: Response,
) -> ElementResponse:
    workspace_id = ensure_workspace(request, response)
    try:
        return ElementResponse.model_validate(update_element(workspace_id, element_id, payload.model_dump(exclude_none=True)))
    except ElementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{element_id}/assets", response_model=ElementResponse)
def add_workspace_element_assets(
    element_id: str,
    request: Request,
    response: Response,
    roles: str = Form(""),
    request_id: str = Form(""),
    files: list[UploadFile] = File(...),
) -> ElementResponse:
    workspace_id = ensure_workspace(request, response)
    try:
        return ElementResponse.model_validate(
            add_element_assets(workspace_id, element_id, _read_uploads(files, roles=roles), request_id=request_id or None)
        )
    except ElementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{element_id}", response_model=ElementResponse)
def delete_workspace_element(element_id: str, request: Request, response: Response) -> ElementResponse:
    workspace_id = ensure_workspace(request, response)
    try:
        return ElementResponse.model_validate(archive_element(workspace_id, element_id))
    except ElementError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
