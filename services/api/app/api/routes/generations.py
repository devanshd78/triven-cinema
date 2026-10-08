import logging
import tempfile
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.generation import (
    AsyncVideoGenerationResponse,
    CombineScenesRequest,
    CombineScenesResponse,
    FullVideoGenerationRequest,
    FullVideoGenerationResponse,
    GenerationCapabilitiesResponse,
    GenerationJobResponse,
    MediaInfo,
    MetricsSummaryResponse,
    ScenePlanRequest,
    ScenePlanResponse,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from app.services.billing_service import (
    InsufficientCreditsError,
    consume_credits,
    refund_credits,
)
from app.services.continuity_service import compose_continuity_prompt, compose_render_integrity_prompt, safe_continuity_id
from app.services.continuity_qc import evaluate_scene_cardinality
from app.services.delivery_service import prepare_delivery, quality_note
from app.services.element_service import (
    ElementError,
    build_reference_sheet,
    canonical_reference_paths as element_canonical_reference_paths,
    compile_element_prompt,
    elements_for_scene,
    resolve_element_bindings,
)
from app.services.identity_service import ensure_workspace, workspace_id_from_request
from app.services.job_service import (
    JobQueueFullError,
    get_job,
    submit_job,
    update_job,
    workspace_owns_generated_file,
)
from app.services.long_render_service import render_long_clip
from app.services.media_probe import probe_media
from app.services.metrics_service import (
    estimate_gpu_cost,
    record_generation_metric,
    summarize_generation_metrics,
)
from app.services.prompt_quality import evaluate_plan_prompt_coverage
from app.services.scene_planner import create_scene_plan
from app.services.storage_service import ensure_minimum_free_disk, resolve_generated_asset
from app.services.video_combiner import combine_videos, extract_continuity_frame
from app.services.video_profiles import (
    duration_profile,
    source_render_dimensions,
    validate_scene_duration,
)
from inference.providers.router import get_video_provider


router = APIRouter()
LOGGER = logging.getLogger("triven.generations")

PROJECT_ROOT = Path(__file__).resolve().parents[5]
GENERATED_DIR = (PROJECT_ROOT / "storage" / "generated").resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

ProgressCallback = Callable[[str, int, str], None]


def media_url(filename: str) -> str:
    return f"/media/generated/{filename}"


def download_url(filename: str) -> str:
    return f"/api/v1/generations/download/{filename}"


def resolve_generated_video(video_url: str) -> Path:
    parsed = urlparse(video_url)
    filename = Path(parsed.path).name

    if not filename or not filename.lower().endswith(".mp4"):
        raise ValueError(f"Invalid generated MP4 URL: {video_url}")

    path = (GENERATED_DIR / filename).resolve()
    if path.parent != GENERATED_DIR:
        raise ValueError("Invalid generated video path.")
    if not path.exists():
        raise FileNotFoundError(f"Generated scene not found: {filename}")
    return path


def _media_info(path: Path) -> MediaInfo:
    return MediaInfo.model_validate(probe_media(path))


def _audio_prompt(prompt: str, audio_direction: str | None) -> str:
    direction = (audio_direction or "").strip()
    if not direction:
        return prompt
    return (
        f"{prompt}\n\n[AUDIO DIRECTION] Generate synchronized audio that follows the visible action. "
        f"{direction} Do not invent spoken dialogue unless it is explicitly requested."
    )


def _generate_video_impl(
    request: VideoGenerationRequest,
    progress: ProgressCallback | None = None,
    *,
    workspace_id: str | None = None,
) -> VideoGenerationResponse:
    validate_scene_duration(
        quality=request.quality,
        duration_seconds=request.duration_seconds,
    )
    if progress:
        progress("initializing", 8, "Checking storage, continuity state and render provider...")

    ensure_minimum_free_disk()
    if request.element_bindings and not workspace_id:
        raise ValueError("Element-bound generation requires a workspace session.")
    try:
        resolved_elements = resolve_element_bindings(workspace_id or "", request.element_bindings) if request.element_bindings else []
    except ElementError as exc:
        raise ValueError(str(exc)) from exc
    active_elements = elements_for_scene(request.prompt, resolved_elements)
    if active_elements and (request.provider or settings.video_provider) != "modal":
        raise ValueError("Reusable Elements currently require the Modal LTX-2.5 provider.")

    provider_key = request.provider or settings.video_provider
    provider = get_video_provider(provider_key, model=request.model)
    render_mode = "dfr" if request.quality != "preview" and provider_key == "modal" else "distilled"
    effective_decoder = "diffusion" if render_mode == "dfr" else request.decoder
    width, height = source_render_dimensions(request.aspect_ratio, request.quality)

    prompt_base = request.prompt
    provider_enhance_prompt = request.enhance_prompt
    enhanced_entity_locks = request.entity_locks
    enhanced_visible_counts = request.visible_entity_counts
    enhanced_character_bible = request.character_bible
    enhanced_style_bible = request.style_bible
    if request.enhance_prompt and provider_key == "modal":
        enhanced = create_scene_plan(
            prompt=request.prompt,
            scene_count=1,
            aspect_ratio=request.aspect_ratio,
            force_ai=True,
        )
        prompt_base = enhanced.scenes[0].prompt
        enhanced_entity_locks = enhanced.entity_locks or enhanced_entity_locks
        enhanced_visible_counts = enhanced.scenes[0].visible_entity_counts or enhanced_visible_counts
        enhanced_character_bible = enhanced.character_bible or enhanced_character_bible
        enhanced_style_bible = enhanced.style_bible or enhanced_style_bible
        provider_enhance_prompt = False

    reference_path: Path | None = None
    if request.continuity_mode == "strict" and request.reference_frame_filename:
        if provider_key != "modal":
            raise ValueError("Strict first-frame continuity currently requires the Modal LTX-2.5 provider.")
        reference_path = resolve_generated_asset(
            request.reference_frame_filename,
            extensions={".png", ".jpg", ".jpeg", ".webp"},
        )

    identity_elements = [item for item in active_elements if item.reference_mode == "identity"]
    start_frame_elements = [item for item in active_elements if item.reference_mode == "start_frame"]
    character_identity_elements = [item for item in identity_elements if item.type == "character"]
    prompt_wardrobe_authoritative = any(
        item.wardrobe_policy == "prompt" for item in character_identity_elements
    )
    use_ingredients = bool(identity_elements or len(active_elements) > 1)
    if use_ingredients and not settings.element_ingredients_enabled:
        raise ValueError("Element identity conditioning is disabled on this deployment.")
    if use_ingredients and request.duration_seconds > settings.element_ingredients_max_scene_seconds + 1e-6:
        raise ValueError(
            f"Element identity-conditioned clips are currently limited to {settings.element_ingredients_max_scene_seconds:g}s."
        )
    if start_frame_elements:
        reference_path = Path(start_frame_elements[0].primary_asset_path)

    element_sheet_path: Path | None = None
    if use_ingredients:
        element_sheet_path = Path(tempfile.gettempdir()) / f"triven-element-sheet-{uuid.uuid4().hex}.png"
        build_reference_sheet(active_elements, element_sheet_path)

    character_detail_element = next(
        (item for item in active_elements if item.type == "character" and item.reference_mode == "identity"),
        None,
    )
    if request.realism_profile == "identity_max" and character_detail_element is None:
        raise ValueError("Identity Max requires an identity-mode Character Element with a real reference image.")
    if progress:
        progress(
            "rendering",
            20,
            "Rendering with LTX-2.5" + (" using the canonical previous-scene anchor..." if reference_path else "..."),
        )

    def chunk_progress(part: int, count: int, message: str) -> None:
        if progress:
            progress(
                "rendering",
                min(76, 20 + int(((part + 1) / max(1, count)) * 56)),
                message,
            )

    total_render_seconds = 0.0
    total_wall_seconds = 0.0
    total_chunks = 0
    continuity_regenerations = 0
    continuity_warnings: list[str] = []
    qc_attempted = False
    qc_unavailable = False
    final_qc_passed = True
    last_qc_note = ""
    result = None
    source_path: Path | None = None

    for attempt in range(max(0, int(request.continuity_max_retries)) + 1):
        prompt_to_render = prompt_base
        if request.continuity_mode != "off":
            prompt_to_render = compose_continuity_prompt(
                scene_prompt=prompt_to_render,
                character_bible=enhanced_character_bible,
                style_bible=enhanced_style_bible,
                scene_index=request.scene_index,
                scene_count=request.scene_count,
                entity_locks=enhanced_entity_locks,
                visible_entity_counts=enhanced_visible_counts,
                reference_frame_present=reference_path is not None,
                prompt_wardrobe_authoritative=prompt_wardrobe_authoritative,
                retry_level=attempt,
                qc_feedback=last_qc_note,
            )
        prompt_to_render = compose_render_integrity_prompt(
            prompt_to_render,
            realism_profile=request.realism_profile,
            prompt_wardrobe_authoritative=prompt_wardrobe_authoritative,
            retry_level=attempt,
            qc_feedback=last_qc_note,
        )
        prompt_to_render = _audio_prompt(prompt_to_render, request.audio_direction)
        if active_elements:
            prompt_to_render = compile_element_prompt(prompt_to_render, active_elements)

        result = render_long_clip(
            provider=provider,
            prompt=prompt_to_render,
            width=width,
            height=height,
            duration_seconds=request.duration_seconds,
            seed=request.seed,
            decoder=effective_decoder,
            enhance_prompt=provider_enhance_prompt if attempt == 0 else False,
            render_mode=render_mode,
            reference_image_path=str(reference_path) if reference_path else None,
            reference_strength=(
                1.0
                if request.realism_profile == "identity_max" and character_identity_elements
                else max(
                    request.continuity_strength,
                    0.95 if character_identity_elements and request.continuity_mode == "strict" else request.continuity_strength,
                )
            ),
            element_reference_sheet_path=str(element_sheet_path) if element_sheet_path else None,
            element_reference_strength=(
                max((item.strength for item in active_elements), default=settings.element_ingredients_strength)
                if use_ingredients else settings.element_ingredients_strength
            ),
            realism_profile=request.realism_profile,
            progress=chunk_progress,
        )
        candidate_path = Path(result.path)
        total_render_seconds += float(result.render_seconds)
        total_wall_seconds += float(result.wall_seconds or result.render_seconds)
        total_chunks += int(result.chunk_count or 1)

        qc = None
        # Direct generation must be inspectable too, even when continuity
        # prompting / previous-frame chaining is off. QC is post-render only.
        if request.continuity_qc_mode != "off":
            if progress:
                progress("rendering", 78, "Checking entity count and duplicate-subject continuity...")
            qc = evaluate_scene_cardinality(
                candidate_path,
                entity_locks=enhanced_entity_locks,
                visible_entity_counts=enhanced_visible_counts,
                character_bible=enhanced_character_bible,
                scene_prompt=prompt_to_render,
                qc_mode=request.continuity_qc_mode,
                reference_frame_path=reference_path,
                canonical_reference_paths=element_canonical_reference_paths(active_elements),
            )
            qc_attempted = qc_attempted or not qc.skipped
            qc_unavailable = qc_unavailable or qc.skipped
            if qc.skipped and request.continuity_qc_mode == "strict" and not settings.factory_qc_fail_open_on_unavailable:
                candidate_path.unlink(missing_ok=True)
                raise RuntimeError(f"Strict continuity QC could not run: {qc.note}")
            if qc.skipped and qc.note:
                continuity_warnings.append(qc.note)

        if qc is None or qc.skipped or (qc.passed and not qc.duplicate_detected):
            source_path = candidate_path
            break

        last_qc_note = qc.note or "; ".join(qc.violations) or "duplicate/cardinality violation"
        if attempt < request.continuity_max_retries:
            continuity_regenerations += 1
            candidate_path.unlink(missing_ok=True)
            continue

        final_qc_passed = False
        message = f"Continuity QC failed after {attempt + 1} attempt(s): {last_qc_note}"
        if request.continuity_qc_mode == "strict" and not settings.factory_preserve_on_qc_failure:
            candidate_path.unlink(missing_ok=True)
            raise RuntimeError(message)
        continuity_warnings.append(message + " Video retained for review; visual QC did not pass.")
        source_path = candidate_path
        break

    if result is None or source_path is None:
        raise RuntimeError("Video render did not produce an accepted scene.")

    continuity_frame_path: Path | None = None
    if request.continuity_mode != "off":
        continuity_tag = safe_continuity_id(request.continuity_id or uuid.uuid4().hex[:12])
        continuity_frame_path = GENERATED_DIR / (
            f"continuity-{continuity_tag}-scene-{(request.scene_index or 0) + 1}-{uuid.uuid4().hex[:8]}.png"
        )
        extract_continuity_frame(source_path, continuity_frame_path)

    if progress:
        progress("delivery", 82, f"Preparing {request.quality} delivery and {request.audio_mode} audio...")

    delivery_path = prepare_delivery(
        source_path,
        aspect_ratio=request.aspect_ratio,
        quality=request.quality,
        audio_mode=request.audio_mode,
        prefix="scene",
    )

    if progress:
        progress("probing", 92, "Validating dimensions, audio and continuity frame...")

    media_info = _media_info(delivery_path)
    wall_seconds = total_wall_seconds
    estimated_cost, estimated_cost_per_minute, cost_note = estimate_gpu_cost(
        render_seconds=wall_seconds,
        gpu=result.gpu,
        output_duration_seconds=request.duration_seconds,
    )

    filename = delivery_path.name
    visual_qc_status = (
        "failed" if qc_attempted and not final_qc_passed else
        "unavailable" if qc_unavailable else
        "passed" if qc_attempted else "not_checked"
    )
    record_generation_metric(
        {
            "type": "scene",
            "provider": result.provider,
            "model": request.model,
            "gpu": result.gpu,
            "aspect_ratio": request.aspect_ratio,
            "source_width": width,
            "source_height": height,
            "delivery_width": media_info.width,
            "delivery_height": media_info.height,
            "duration_seconds": request.duration_seconds,
            "chunk_count": total_chunks,
            "render_seconds": round(total_render_seconds, 3),
            "wall_seconds": round(wall_seconds, 3),
            "quality": request.quality,
            "audio_mode": request.audio_mode,
            "decoder": effective_decoder,
            "render_mode": render_mode,
            "realism_profile": request.realism_profile,
            "detail_refined": result.detail_refined,
            "seed": request.seed,
            "has_audio": media_info.has_audio,
            "audio_codec": media_info.audio_codec,
            "continuity_mode": request.continuity_mode,
            "continuity_id": request.continuity_id,
            "continuity_applied": result.reference_conditioned,
            "reference_frame_filename": request.reference_frame_filename,
            "continuity_frame_filename": continuity_frame_path.name if continuity_frame_path else None,
            "entity_lock_count": len(enhanced_entity_locks),
            "continuity_qc_mode": request.continuity_qc_mode,
            "continuity_qc_passed": (final_qc_passed if qc_attempted else None),
            "visual_qc_status": visual_qc_status,
            "continuity_regenerations": continuity_regenerations,
            "estimated_cost_usd": estimated_cost,
            "estimated_cost_per_output_minute_usd": estimated_cost_per_minute,
            "filename": filename,
        }
    )

    # Delivery masters replace their source clip for the public response. Keep
    # only the continuity PNG when needed, reducing VPS disk pressure.
    if delivery_path != source_path:
        source_path.unlink(missing_ok=True)

    if element_sheet_path is not None:
        element_sheet_path.unlink(missing_ok=True)

    return VideoGenerationResponse(
        video_url=media_url(filename),
        download_url=download_url(filename),
        filename=filename,
        seed=result.seed,
        render_details=result.render_details,
        render_seconds=round(total_render_seconds, 2),
        wall_seconds=round(wall_seconds, 2),
        provider=result.provider,
        model=request.model,
        quality=request.quality,
        quality_note=quality_note(request.quality, request.aspect_ratio),
        realism_profile=request.realism_profile,
        detail_refined=result.detail_refined,
        audio_mode=request.audio_mode,
        chunk_count=total_chunks,
        gpu=result.gpu,
        media_info=media_info,
        estimated_cost_usd=estimated_cost,
        estimated_cost_per_output_minute_usd=estimated_cost_per_minute,
        cost_note=cost_note,
        continuity_mode=request.continuity_mode,
        continuity_applied=result.reference_conditioned,
        reference_frame_filename=request.reference_frame_filename,
        continuity_frame_url=media_url(continuity_frame_path.name) if continuity_frame_path else None,
        continuity_frame_filename=continuity_frame_path.name if continuity_frame_path else None,
        continuity_qc_passed=(final_qc_passed if qc_attempted else None),
        visual_qc_status=visual_qc_status,
        continuity_regenerations=continuity_regenerations,
        continuity_warnings=continuity_warnings,
        elements_used=[f"@{item.handle}" for item in active_elements],
        element_reference_mode=(
            "ingredients" if use_ingredients else ("start_frame" if start_frame_elements else None)
        ),
    )


@router.get("/capabilities", response_model=GenerationCapabilitiesResponse)
async def generation_capabilities():
    cost_tracking_configured = any(
        rate > 0
        for rate in (
            settings.modal_gpu_hourly_usd_b200,
            settings.modal_gpu_hourly_usd_h200,
            settings.modal_gpu_hourly_usd_h100,
        )
    )
    durations = duration_profile()
    return GenerationCapabilitiesResponse(
        providers=[
            {"id": "huggingface", "label": "ZeroGPU (development)", "available": True, "paid": False},
            {"id": "modal", "label": "Modal self-hosted LTX-2.5", "available": True, "paid": True},
        ],
        models=[
            {"id": "ltx-2.5", "label": "LTX 2.5", "available": True},
            {"id": "wan", "label": "WAN", "available": False},
            {"id": "minimax", "label": "MiniMax", "available": False},
        ],
        qualities=[
            {
                "id": "preview",
                "label": "Source preview",
                "description": "LTX-2.5 preview profile; Factory scenes use 15-20 second single-pass shots.",
            },
            {
                "id": "1080p",
                "label": "1080p master",
                "description": "Production LTX-2.5 DFR source render with diffusion decode; Factory uses 15-20s single-pass shots, with an optional experimental 30s B200 single-pass shot.",
            },
            {
                "id": "4k",
                "label": "4K delivery master",
                "description": "Production LTX-2.5 DFR source render with diffusion decode, followed by the validated 4K delivery master; not a native-4K source claim.",
            },
        ],
        aspect_ratios=["16:9", "9:16", "1:1"],
        decoders=["conv", "diffusion"],
        realism_profiles=[
            {"id": "standard", "label": "Standard", "description": "Existing production path without the extra texture refinement pass."},
            {"id": "real_skin", "label": "Real Skin", "description": "Final-quality DFR/Ingredients render plus the official tiled LTX-2.5 Refine Details IC-LoRA."},
            {"id": "identity_max", "label": "Identity Max", "description": "Real Skin final path intended for a character Element made from a sharp real reference photo."},
        ],
        continuity_modes=["off", "balanced", "strict"],
        audio_modes=["native", "mastered", "mute"],
        image_conditioning=True,
        entity_count_lock=True,
        continuity_vision_qc=settings.continuity_vision_qc_enabled,
        max_scene_duration_seconds=max(durations.values()),
        max_scene_duration_seconds_by_quality=durations,
        native_chunk_seconds=settings.ltx_native_chunk_seconds,
        max_factory_duration_seconds=settings.max_factory_duration_seconds,
        async_jobs=True,
        audio_probe=True,
        cost_tracking_configured=cost_tracking_configured,
        gpu=settings.triven_modal_gpu,
        production_mode=settings.is_production,
        job_workers=settings.job_workers,
        job_max_pending=settings.job_max_pending,
        elements={
            "enabled": True,
            "ingredients_enabled": settings.element_ingredients_enabled,
            "max_stored": settings.element_max_stored_per_workspace,
            "max_assets_per_element": settings.element_max_assets_per_element,
            "max_active_per_scene": settings.element_max_active_per_scene,
            "max_characters_per_scene": settings.element_max_characters_per_scene,
            "max_props_per_scene": settings.element_max_props_per_scene,
            "max_locations_per_scene": settings.element_max_locations_per_scene,
            "max_styles_per_scene": settings.element_max_styles_per_scene,
            "ingredients_max_scene_seconds": settings.element_ingredients_max_scene_seconds,
        },
    )


@router.post("/plan", response_model=ScenePlanResponse)
def plan_generation(request: ScenePlanRequest):
    try:
        plan = create_scene_plan(
            prompt=request.prompt,
            scene_count=request.scene_count,
            aspect_ratio=request.aspect_ratio,
        )
        plan_quality = evaluate_plan_prompt_coverage(
            request.prompt,
            [scene.prompt for scene in plan.scenes],
        )
        return ScenePlanResponse(
            original_prompt=request.prompt,
            aspect_ratio=request.aspect_ratio,
            scenes=plan.scenes,
            plan_quality=plan_quality,
            planner_source=plan.source,
            planner_note=plan.note,
            continuity_id=f"story-{uuid.uuid4().hex[:16]}",
            character_bible=plan.character_bible,
            style_bible=plan.style_bible,
            entity_locks=plan.entity_locks,
        )
    except Exception as exc:
        LOGGER.exception("Scene planning failed")
        detail = str(exc) if settings.debug and not settings.is_production else "Scene planning failed."
        raise HTTPException(status_code=500, detail=detail) from exc


@router.post("/video", response_model=VideoGenerationResponse)
def generate_video(payload: VideoGenerationRequest, request: Request, response: Response):
    """Synchronous compatibility endpoint. Prefer /jobs/video in the UI."""
    if settings.is_production and not settings.enable_sync_render_endpoints:
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        workspace_id = ensure_workspace(request, response)
        return _generate_video_impl(payload, workspace_id=workspace_id)
    except Exception as exc:
        LOGGER.exception("Video generation failed")
        detail = str(exc) if settings.debug and not settings.is_production else "Video generation failed."
        raise HTTPException(status_code=500, detail=detail) from exc


@router.post("/jobs/video", response_model=AsyncVideoGenerationResponse)
def create_video_job(
    payload: VideoGenerationRequest,
    request: Request,
    response: Response,
):
    workspace_id = ensure_workspace(request, response)
    charge_seconds = max(1, int(round(payload.duration_seconds)))
    charge_reference = f"scene:{uuid.uuid4().hex}"
    try:
        consume_credits(workspace_id, charge_seconds, charge_reference)
    except InsufficientCreditsError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc

    job_payload = payload.model_dump()
    job_payload["workspace_id"] = workspace_id

    def runner(job_id: str) -> dict:
        def progress(stage: str, percent: int, message: str) -> None:
            update_job(job_id, status="running", stage=stage, progress=percent, message=message)

        try:
            result = _generate_video_impl(payload, progress=progress, workspace_id=workspace_id)
            return result.model_dump()
        except Exception:
            refund_credits(workspace_id, charge_seconds, charge_reference)
            raise

    try:
        job_id = submit_job("video", job_payload, runner)
    except JobQueueFullError as exc:
        refund_credits(workspace_id, charge_seconds, charge_reference)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception:
        refund_credits(workspace_id, charge_seconds, charge_reference)
        raise

    return AsyncVideoGenerationResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/api/v1/generations/jobs/{job_id}",
    )


@router.get("/jobs/{job_id}", response_model=GenerationJobResponse)
async def generation_job_status(
    job_id: str,
    request: Request,
    response: Response,
):
    workspace_id = ensure_workspace(request, response)
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found.")

    # Paid/background jobs are scoped to the signed browser workspace. Returning
    # 404 for another workspace avoids leaking render prompts, progress or outputs.
    owner = str((job.get("payload") or {}).get("workspace_id") or "")
    if owner and owner != workspace_id:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    if settings.is_production and not owner:
        raise HTTPException(status_code=404, detail="Generation job not found.")

    return GenerationJobResponse.model_validate(job)


@router.post("/combine", response_model=CombineScenesResponse)
def combine_existing_scenes(request: CombineScenesRequest):
    if settings.is_production and not settings.enable_sync_render_endpoints:
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        ensure_minimum_free_disk()
        video_paths = [resolve_generated_video(url) for url in request.scene_video_urls]
        combined_path = GENERATED_DIR / f"final-source-{uuid.uuid4().hex}.mp4"
        combine_videos(video_paths, combined_path)
        delivery_path = prepare_delivery(
            combined_path,
            aspect_ratio=request.aspect_ratio,
            quality=request.quality,
            audio_mode=request.audio_mode,
            prefix="final",
        )
        media_info = _media_info(delivery_path)
        filename = delivery_path.name
        if delivery_path != combined_path:
            combined_path.unlink(missing_ok=True)

        record_generation_metric(
            {
                "type": "combine",
                "scene_count": len(video_paths),
                "aspect_ratio": request.aspect_ratio,
                "quality": request.quality,
                "audio_mode": request.audio_mode,
                "delivery_width": media_info.width,
                "delivery_height": media_info.height,
                "has_audio": media_info.has_audio,
                "audio_codec": media_info.audio_codec,
                "filename": filename,
            }
        )
        return CombineScenesResponse(
            final_video_url=media_url(filename),
            final_download_url=download_url(filename),
            final_filename=filename,
            scene_count=len(video_paths),
            quality=request.quality,
            quality_note=quality_note(request.quality, request.aspect_ratio),
            audio_mode=request.audio_mode,
            media_info=media_info,
        )
    except Exception as exc:
        LOGGER.exception("Video combine failed")
        detail = str(exc) if settings.debug and not settings.is_production else "Video combine failed."
        raise HTTPException(status_code=500, detail=detail) from exc


@router.post("/full-video", response_model=FullVideoGenerationResponse)
def generate_full_video(request: FullVideoGenerationRequest):
    """Development compatibility route. Production uses async scene/factory jobs."""
    if settings.is_production and not settings.enable_sync_render_endpoints:
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        validate_scene_duration(quality=request.quality, duration_seconds=request.duration_seconds)
        ensure_minimum_free_disk()
        provider_key = request.provider or settings.video_provider
        provider = get_video_provider(provider_key, model=request.model)
        render_mode = "dfr" if request.quality != "preview" and provider_key == "modal" else "distilled"
        effective_decoder = "diffusion" if render_mode == "dfr" else request.decoder
        width, height = source_render_dimensions(request.aspect_ratio, request.quality)

        generated_paths: list[Path] = []
        scene_urls: list[str] = []
        render_details: list[str] = []
        total_render_seconds = 0.0
        total_wall_seconds = 0.0
        gpu: str | None = None
        previous_frame: Path | None = None
        continuity_frames: list[Path] = []

        try:
            for index, scene in enumerate(request.scenes):
                locked_prompt = (
                    compose_continuity_prompt(
                        scene_prompt=scene.prompt,
                        character_bible=request.character_bible,
                        style_bible=request.style_bible,
                        scene_index=index,
                        scene_count=len(request.scenes),
                        entity_locks=request.entity_locks,
                        visible_entity_counts=scene.visible_entity_counts,
                        reference_frame_present=(request.continuity_mode == "strict" and previous_frame is not None),
                    )
                    if request.continuity_mode != "off"
                    else scene.prompt
                )
                locked_prompt = compose_render_integrity_prompt(
                    locked_prompt,
                    realism_profile=request.realism_profile,
                    prompt_wardrobe_authoritative=False,
                )
                locked_prompt = _audio_prompt(locked_prompt, request.audio_direction)
                result = render_long_clip(
                    provider=provider,
                    prompt=locked_prompt,
                    width=width,
                    height=height,
                    duration_seconds=request.duration_seconds,
                    seed=request.seed,
                    decoder=effective_decoder,
                    enhance_prompt=request.enhance_prompt and index == 0,
                    render_mode=render_mode,
                    realism_profile=request.realism_profile,
                    reference_image_path=(
                        str(previous_frame)
                        if request.continuity_mode == "strict" and previous_frame
                        else None
                    ),
                    reference_strength=request.continuity_strength,
                )
                clip_path = Path(result.path)
                generated_paths.append(clip_path)
                scene_urls.append(media_url(clip_path.name))
                render_details.append(result.render_details)
                total_render_seconds += result.render_seconds
                total_wall_seconds += float(result.wall_seconds or result.render_seconds)
                gpu = result.gpu or gpu
                if request.continuity_mode != "off" and index < len(request.scenes) - 1:
                    previous_frame = GENERATED_DIR / f".continuity-full-{uuid.uuid4().hex}.png"
                    extract_continuity_frame(clip_path, previous_frame)
                    continuity_frames.append(previous_frame)

            combined_path = GENERATED_DIR / f"final-source-{uuid.uuid4().hex}.mp4"
            combine_videos(generated_paths, combined_path)
            delivery_path = prepare_delivery(
                combined_path,
                aspect_ratio=request.aspect_ratio,
                quality=request.quality,
                audio_mode=request.audio_mode,
                prefix="final",
            )
            media_info = _media_info(delivery_path)
            filename = delivery_path.name
            output_duration = request.duration_seconds * len(request.scenes)
            estimated_cost, estimated_cost_per_minute, cost_note = estimate_gpu_cost(
                render_seconds=total_wall_seconds,
                gpu=gpu,
                output_duration_seconds=output_duration,
            )
            if delivery_path != combined_path:
                combined_path.unlink(missing_ok=True)

            record_generation_metric(
                {
                    "type": "full-video",
                    "provider": provider.name,
                    "model": request.model,
                    "gpu": gpu,
                    "aspect_ratio": request.aspect_ratio,
                    "scene_count": len(request.scenes),
                    "duration_seconds_per_scene": request.duration_seconds,
                    "render_seconds": round(total_render_seconds, 3),
                    "wall_seconds": round(total_wall_seconds, 3),
                    "quality": request.quality,
                    "audio_mode": request.audio_mode,
                    "decoder": effective_decoder,
                    "render_mode": render_mode,
                    "realism_profile": request.realism_profile,
                    "detail_refined": any("Refine Details IC-LoRA" in detail for detail in render_details),
                    "has_audio": media_info.has_audio,
                    "audio_codec": media_info.audio_codec,
                    "estimated_cost_usd": estimated_cost,
                    "estimated_cost_per_output_minute_usd": estimated_cost_per_minute,
                    "filename": filename,
                }
            )
            return FullVideoGenerationResponse(
                final_video_url=media_url(filename),
                final_download_url=download_url(filename),
                final_filename=filename,
                scene_video_urls=scene_urls,
                render_details=render_details,
                total_render_seconds=round(total_render_seconds, 2),
                total_wall_seconds=round(total_wall_seconds, 2),
                provider=provider.name,
                model=request.model,
                quality=request.quality,
                quality_note=quality_note(request.quality, request.aspect_ratio),
                realism_profile=request.realism_profile,
                detail_refined=any("Refine Details IC-LoRA" in detail for detail in render_details),
                audio_mode=request.audio_mode,
                gpu=gpu,
                media_info=media_info,
                estimated_cost_usd=estimated_cost,
                estimated_cost_per_output_minute_usd=estimated_cost_per_minute,
                cost_note=cost_note,
            )
        finally:
            for frame in continuity_frames:
                frame.unlink(missing_ok=True)
    except Exception as exc:
        LOGGER.exception("Full video generation failed")
        detail = str(exc) if settings.debug and not settings.is_production else "Full video generation failed."
        raise HTTPException(status_code=500, detail=detail) from exc


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def metrics_summary():
    if settings.is_production and not settings.enable_metrics_endpoint:
        raise HTTPException(status_code=404, detail="Not found.")
    return MetricsSummaryResponse.model_validate(summarize_generation_metrics())


@router.get("/download/{filename}")
async def download_generated_video(filename: str, request: Request):
    try:
        path = resolve_generated_video(filename)
        if settings.is_production:
            workspace_id = workspace_id_from_request(request)
            if not workspace_id or not workspace_owns_generated_file(workspace_id, path.name):
                raise FileNotFoundError("Generated video not found.")
        return FileResponse(path, media_type="video/mp4", filename=path.name)
    except Exception as exc:
        detail = str(exc) if settings.debug and not settings.is_production else "Generated video not found."
        raise HTTPException(status_code=404, detail=detail) from exc
