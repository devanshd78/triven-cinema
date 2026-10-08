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
    AudioRetakeRequest,
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
from app.services.continuity_qc import ContinuityQCResult, evaluate_scene_cardinality
from app.services.delivery_service import prepare_delivery, quality_note
from app.services.element_service import (
    ElementError,
    build_reference_sheet,
    canonical_reference_paths as element_canonical_reference_paths,
    compile_element_prompt,
    elements_for_scene,
    resolve_element_bindings,
    validate_prompt_bindings,
)
from app.services.identity_service import ensure_workspace, workspace_id_from_request
from app.services.job_service import (
    JobQueueFullError,
    get_job,
    submit_job,
    update_job,
    checkpoint_job_result,
    list_jobs,
    find_job_by_request_id,
    generated_asset_metadata,
    workspace_owns_generated_file,
)
from app.services.job_request_service import submit_generation_request
from app.services.long_render_service import render_long_clip
from app.services.dialogue_service import build_audio_prompt, separate_visual_and_script
from app.services.render_review_service import aggregate_qc, pass_flag, qc_report, register_render, review_audio
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
        validate_prompt_bindings(request.prompt, resolved_elements)
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

    prompt_base, spoken_script = separate_visual_and_script(request.prompt, request.spoken_script)
    provider_enhance_prompt = request.enhance_prompt
    enhanced_entity_locks = request.entity_locks
    enhanced_visible_counts = request.visible_entity_counts
    enhanced_character_bible = request.character_bible
    enhanced_style_bible = request.style_bible
    if request.enhance_prompt and provider_key == "modal":
        enhanced = create_scene_plan(
            prompt=prompt_base,
            spoken_script=spoken_script,
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
        if workspace_id and (settings.auth_enabled or settings.is_production) and not workspace_owns_generated_file(workspace_id, request.reference_frame_filename):
            raise ValueError("Continuity reference not found in this account.")
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
    last_qc_note = ""
    result = None
    source_path: Path | None = None
    visual_report = qc_report()
    scene_attempts: list[dict] = []
    delivery_warnings: list[str] = []

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
        if active_elements:
            prompt_to_render = compile_element_prompt(prompt_to_render, active_elements)
        prompt_to_render = build_audio_prompt(prompt_to_render, spoken_script, request.audio_direction)

        try:
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
            candidate = Path(result.path)
            if not candidate.is_file() or not candidate.stat().st_size:
                raise RuntimeError("The inference provider produced no playable video file.")
        except Exception as exc:
            if source_path is None:
                raise
            continuity_warnings.append(f"Regeneration unavailable ({type(exc).__name__}); previous rendered video retained for review.")
            break
        candidate_path = Path(result.path)
        source_path = candidate_path
        register_render(workspace_id, source_path, kind="raw", scene_index=request.scene_index or 0,
                        attempt=attempt, visual_qc_status="not_checked", audio_qc_status="not_checked")
        total_render_seconds += float(result.render_seconds)
        total_wall_seconds += float(result.wall_seconds or result.render_seconds)
        total_chunks += int(result.chunk_count or 1)

        qc = None
        if request.continuity_qc_mode != "off":
            if progress:
                progress("rendering", 78, "Inspecting visual identity and artifacts...")
            try:
                qc = evaluate_scene_cardinality(
                    candidate_path, entity_locks=enhanced_entity_locks,
                    visible_entity_counts=enhanced_visible_counts,
                    character_bible=enhanced_character_bible, scene_prompt=prompt_to_render,
                    qc_mode=request.continuity_qc_mode, reference_frame_path=reference_path,
                    canonical_reference_paths=element_canonical_reference_paths(active_elements),
                )
            except Exception as exc:
                qc = ContinuityQCResult(skipped=True, note=f"Visual QC unavailable ({type(exc).__name__}).")
        visual_report = qc_report(qc)
        scene_attempts.append({"filename": source_path.name, "attempt": attempt + 1, **visual_report})
        register_render(workspace_id, source_path, kind="scene", scene_index=request.scene_index or 0,
                        visual_qc_status=visual_report["status"], visual_qc=visual_report,
                        visual_qc_attempts=scene_attempts)
        checkpoint_job_result({"filename": source_path.name, "video_url": media_url(source_path.name),
                               "download_url": download_url(source_path.name),
                               "visual_qc_status": visual_report["status"], "visual_qc": visual_report})
        if visual_report["status"] != "failed":
            if visual_report["status"] == "unavailable":
                continuity_warnings.append(visual_report["note"])
            break
        last_qc_note = visual_report.get("note") or "; ".join(visual_report.get("violations") or []) or "Visual quality failure"
        if attempt < request.continuity_max_retries:
            continuity_regenerations += 1
            continue
        continuity_warnings.append(f"Continuity QC failed after {attempt + 1} attempt(s): {last_qc_note}. Video retained for review; visual QC did not pass.")
        break

    if result is None or source_path is None:
        raise RuntimeError("Video render did not produce an accepted scene.")

    audio_review = review_audio(
        source_path, provider=provider, workspace_id=workspace_id, scene_prompt=prompt_base,
        spoken_script=spoken_script, audio_direction=request.audio_direction, audio_mode=request.audio_mode,
        duration_seconds=request.duration_seconds, seed=request.seed, scene_index=request.scene_index or 0,
    )
    source_path = audio_review.path
    if audio_review.retake is not None:
        total_render_seconds += float(audio_review.retake.render_seconds)
        total_wall_seconds += float(audio_review.retake.wall_seconds or audio_review.retake.render_seconds)
        total_chunks += int(audio_review.retake.chunk_count or 1)
    continuity_frame_path: Path | None = None
    if request.continuity_mode != "off" and visual_report["status"] != "failed":
        continuity_tag = safe_continuity_id(request.continuity_id or uuid.uuid4().hex[:12])
        candidate_frame = GENERATED_DIR / f"continuity-{continuity_tag}-scene-{(request.scene_index or 0) + 1}-{uuid.uuid4().hex[:8]}.png"
        try:
            extract_continuity_frame(source_path, candidate_frame)
            register_render(workspace_id, candidate_frame, kind="continuity_frame", scene_index=request.scene_index or 0)
            continuity_frame_path = candidate_frame
        except Exception as exc:
            continuity_warnings.append(f"Continuity anchor unavailable ({type(exc).__name__}); rendered video retained.")
    if progress:
        progress("delivery", 82, f"Preparing {request.quality} delivery and {request.audio_mode} audio...")
    delivery_complete = True
    try:
        delivery_path = prepare_delivery(source_path, aspect_ratio=request.aspect_ratio,
                                         quality=request.quality, audio_mode=request.audio_mode, prefix="scene")
        register_render(workspace_id, delivery_path, kind="delivery", scene_index=request.scene_index or 0)
    except Exception as exc:
        delivery_path = source_path
        delivery_complete = False
        delivery_warnings.append(f"Delivery processing unavailable ({type(exc).__name__}); original rendered video retained.")
    try:
        media_info = _media_info(delivery_path)
    except Exception as exc:
        media_info = MediaInfo()
        delivery_complete = False
        delivery_warnings.append(f"Media inspection unavailable ({type(exc).__name__}); rendered file retained.")
    wall_seconds = total_wall_seconds
    estimated_cost, estimated_cost_per_minute, cost_note = estimate_gpu_cost(
        render_seconds=wall_seconds,
        gpu=result.gpu,
        output_duration_seconds=request.duration_seconds,
    )

    filename = delivery_path.name
    visual_qc_status = visual_report["status"]
    audio_qc_status = audio_review.report["status"]
    scene_record = {"scene_index": request.scene_index or 0, "filename": filename,
                    "video_url": media_url(filename), "download_url": download_url(filename),
                    "spoken_script": spoken_script,
                    "visual_qc_status": visual_qc_status, "visual_qc": visual_report,
                    "audio_qc_status": audio_qc_status, "audio_qc": audio_review.report,
                    "visual_qc_attempts": scene_attempts, "audio_qc_attempts": audio_review.attempts}
    register_render(workspace_id, delivery_path, **scene_record)
    checkpoint_job_result({**scene_record, "scene_results": [scene_record], "warnings": delivery_warnings,
                           "delivery_complete": delivery_complete})
    try:
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
                "continuity_qc_passed": pass_flag(visual_qc_status),
                "visual_qc_status": visual_qc_status,
                "continuity_regenerations": continuity_regenerations,
                "estimated_cost_usd": estimated_cost,
                "estimated_cost_per_output_minute_usd": estimated_cost_per_minute,
                "filename": filename,
            }
        )
    except Exception:
        delivery_warnings.append("Usage metrics could not be recorded; the rendered video is available.")

    if element_sheet_path is not None:
        element_sheet_path.unlink(missing_ok=True)

    response = VideoGenerationResponse(
        audio_qc_passed=pass_flag(audio_qc_status), audio_qc_status=audio_qc_status,
        audio_qc=audio_review.report, visual_qc=visual_report,
        audio_retake_count=int(audio_review.retake is not None), audio_warnings=audio_review.warnings,
        scene_results=[scene_record], warnings=delivery_warnings, delivery_complete=delivery_complete,
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
        quality_note=(quality_note(request.quality, request.aspect_ratio) if delivery_complete else
                      "Original rendered source retained; requested delivery processing did not complete."),
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
        continuity_qc_passed=pass_flag(visual_qc_status),
        visual_qc_status=visual_qc_status,
        continuity_regenerations=continuity_regenerations,
        continuity_warnings=continuity_warnings,
        elements_used=[f"@{item.handle}" for item in active_elements],
        element_reference_mode=(
            "ingredients" if use_ingredients else ("start_frame" if start_frame_elements else None)
        ),
    )

    checkpoint_job_result(response.model_dump())
    return response


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
            "max_upload_mb": settings.element_max_upload_mb,
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
            spoken_script=request.spoken_script,
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
    def runner(job_id: str) -> dict:
        def progress(stage: str, percent: int, message: str) -> None:
            update_job(job_id, status="running", stage=stage, progress=percent, message=message)
        return _generate_video_impl(payload, progress=progress, workspace_id=workspace_id).model_dump()
    return submit_generation_request("video", payload, workspace_id, runner,
                                     charge_seconds=max(1, int(round(payload.duration_seconds))))


@router.get("/jobs", response_model=list[GenerationJobResponse])
def generation_job_list(request: Request, response: Response, chat_id: str | None = None, limit: int = 100):
    workspace_id = ensure_workspace(request, response)
    return list_jobs(workspace_id, chat_id=chat_id, limit=limit)


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


def _owned_video(workspace_id: str, value: str) -> Path:
    path = resolve_generated_video(value)
    if not workspace_owns_generated_file(workspace_id, path.name):
        raise ValueError("Generated video not found in this account.")
    return path


def _combine_impl(payload: CombineScenesRequest, workspace_id: str, progress=None) -> CombineScenesResponse:
    ensure_minimum_free_disk()
    paths = [_owned_video(workspace_id, value) for value in payload.scene_video_urls]
    scenes = []
    for index, path in enumerate(paths):
        metadata = generated_asset_metadata(workspace_id, path.name) or {}
        scenes.append({**metadata, "scene_index": index, "filename": path.name,
                       "video_url": media_url(path.name), "download_url": download_url(path.name),
                       "visual_qc_status": metadata.get("visual_qc_status", "not_checked"),
                       "audio_qc_status": metadata.get("audio_qc_status", "not_checked")})
    checkpoint_job_result({"scene_results": scenes})
    if progress:
        progress("composing", 30, "Combining saved scenes and audio...")
    combined = GENERATED_DIR / f"final-source-{uuid.uuid4().hex}.mp4"
    combine_videos(paths, combined)
    register_render(workspace_id, combined, kind="composed", scene_results=scenes)
    checkpoint_job_result({"final_filename": combined.name, "final_video_url": media_url(combined.name),
                           "final_download_url": download_url(combined.name)})
    warnings = []
    delivery_complete = True
    if progress:
        progress("delivery", 80, "Preparing the delivery master...")
    try:
        delivery = prepare_delivery(combined, aspect_ratio=payload.aspect_ratio, quality=payload.quality,
                                    audio_mode=payload.audio_mode, prefix="final")
        register_render(workspace_id, delivery, kind="delivery")
    except Exception as exc:
        delivery = combined
        delivery_complete = False
        warnings.append(f"Delivery processing unavailable ({type(exc).__name__}); combined source retained.")
    try:
        info = _media_info(delivery)
    except Exception as exc:
        info = MediaInfo()
        delivery_complete = False
        warnings.append(f"Media inspection unavailable ({type(exc).__name__}); combined source retained.")
    visual_status = aggregate_qc([{"status": scene["visual_qc_status"]} for scene in scenes])
    audio_status = "not_checked" if payload.audio_mode == "mute" else aggregate_qc([{"status": scene["audio_qc_status"]} for scene in scenes])
    result = CombineScenesResponse(final_video_url=media_url(delivery.name), final_download_url=download_url(delivery.name),
                                   final_filename=delivery.name, scene_count=len(paths), quality=payload.quality,
                                   quality_note=(quality_note(payload.quality, payload.aspect_ratio) if delivery_complete else
                                                 "Combined source retained; requested delivery processing did not complete."),
                                   delivery_complete=delivery_complete, audio_mode=payload.audio_mode,
                                   media_info=info, visual_qc_status=visual_status, audio_qc_status=audio_status,
                                   scene_results=scenes, warnings=warnings)
    register_render(workspace_id, delivery, **result.model_dump())
    checkpoint_job_result(result.model_dump())
    return result


@router.post("/jobs/combine", response_model=AsyncVideoGenerationResponse)
def create_combine_job(payload: CombineScenesRequest, request: Request, response: Response):
    workspace_id = ensure_workspace(request, response)
    # Authorize every input before queueing CPU work.
    try:
        for value in payload.scene_video_urls:
            _owned_video(workspace_id, value)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Generated scene not found in this account.") from exc
    def runner(job_id: str) -> dict:
        def progress(stage, percent, message):
            update_job(job_id, status="running", stage=stage, progress=percent, message=message)
        return _combine_impl(payload, workspace_id, progress).model_dump()
    return submit_generation_request("combine", payload, workspace_id, runner)


@router.post("/combine", response_model=CombineScenesResponse)
def combine_existing_scenes(payload: CombineScenesRequest, request: Request, response: Response):
    if settings.is_production and not settings.enable_sync_render_endpoints:
        raise HTTPException(status_code=404, detail="Not found.")
    workspace_id = ensure_workspace(request, response)
    return _combine_impl(payload, workspace_id)


@router.post("/jobs/audio-retake", response_model=AsyncVideoGenerationResponse)
def create_audio_retake_job(payload: AudioRetakeRequest, request: Request, response: Response):
    workspace_id = ensure_workspace(request, response)
    try:
        source = _owned_video(workspace_id, payload.filename)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Generated video not found in this account.") from exc

    source_info = _media_info(source)
    if not source_info.duration_seconds:
        raise HTTPException(status_code=422, detail="The saved clip has no readable runtime.")

    def runner(job_id: str) -> dict:
        original = generated_asset_metadata(workspace_id, source.name) or {}
        checkpoint_job_result({"filename": source.name, "video_url": media_url(source.name), "download_url": download_url(source.name)})
        info = _media_info(source)
        if not info.duration_seconds:
            raise ValueError("The saved clip has no readable runtime.")
        provider = get_video_provider("modal", model="ltx-2.5")
        update_job(job_id, status="running", stage="rendering", progress=25, message="Repairing speech while preserving the saved picture...")
        retake = provider.retake_audio(video_path=str(source), prompt=build_audio_prompt("", payload.spoken_script, payload.audio_direction),
                                      duration_seconds=info.duration_seconds, seed=payload.seed)
        repaired = Path(retake.path)
        if not repaired.is_file() or not repaired.stat().st_size:
            raise RuntimeError("Audio repair produced no playable video file; the original remains available.")
        register_render(workspace_id, repaired, kind="audio_retake", source_filename=source.name,
                        visual_qc_status=original.get("visual_qc_status", "not_checked"), audio_qc_status="not_checked")
        review = review_audio(repaired, provider=provider, workspace_id=workspace_id, scene_prompt="",
                              spoken_script=payload.spoken_script, audio_direction=payload.audio_direction,
                              audio_mode="native", duration_seconds=info.duration_seconds, seed=payload.seed,
                              scene_index=original.get("scene_index", 0), allow_retake=False)
        repair_warnings = list(review.warnings)
        try:
            repaired_info = _media_info(repaired)
        except Exception as exc:
            repaired_info = MediaInfo()
            repair_warnings.append(f"Repaired media inspection unavailable ({type(exc).__name__}); saved repair is available for review.")
        result = {"filename": repaired.name, "video_url": media_url(repaired.name), "download_url": download_url(repaired.name),
                  "audio_mode": "native", "spoken_script": payload.spoken_script,
                  "visual_qc_status": original.get("visual_qc_status", "not_checked"), "visual_qc": original.get("visual_qc"),
                  "audio_qc_status": review.report["status"], "audio_qc_passed": pass_flag(review.report["status"]),
                  "audio_qc": review.report, "audio_warnings": review.warnings, "audio_retake_count": 1,
                  "warnings": repair_warnings, "media_info": repaired_info.model_dump()}
        register_render(workspace_id, repaired, **result)
        checkpoint_job_result(result)
        return result
    return submit_generation_request("audio_retake", payload, workspace_id, runner,
                                     charge_seconds=max(1, int(round(source_info.duration_seconds))))


@router.post("/full-video", response_model=FullVideoGenerationResponse)
def generate_full_video(payload: FullVideoGenerationRequest, request: Request, response: Response):
    """Compatibility route using the same owned, inspected scene pipeline as jobs."""
    if settings.is_production and not settings.enable_sync_render_endpoints:
        raise HTTPException(status_code=404, detail="Not found.")
    workspace_id = ensure_workspace(request, response)
    scenes = []
    anchor = None
    for index, scene in enumerate(payload.scenes):
        render_request = VideoGenerationRequest(
            prompt=scene.prompt, spoken_script=scene.spoken_script,
            aspect_ratio=payload.aspect_ratio, duration_seconds=payload.duration_seconds,
            seed=(payload.seed + index * 104729) % 2147483648, decoder=payload.decoder,
            enhance_prompt=payload.enhance_prompt, quality=payload.quality,
            realism_profile=payload.realism_profile, audio_mode=payload.audio_mode,
            audio_direction=payload.audio_direction, provider=payload.provider, model=payload.model,
            continuity_mode=payload.continuity_mode, continuity_id=payload.continuity_id,
            scene_index=index, scene_count=len(payload.scenes), character_bible=payload.character_bible,
            style_bible=payload.style_bible, entity_locks=payload.entity_locks,
            visible_entity_counts=scene.visible_entity_counts, continuity_qc_mode=payload.continuity_qc_mode,
            continuity_max_retries=payload.continuity_max_retries, continuity_strength=payload.continuity_strength,
            element_bindings=payload.element_bindings, reference_frame_filename=anchor,
        )
        rendered = _generate_video_impl(render_request, workspace_id=workspace_id)
        scenes.append(rendered)
        anchor = rendered.continuity_frame_filename if rendered.visual_qc_status != "failed" else None
    combined = _combine_impl(CombineScenesRequest(scene_video_urls=[scene.video_url for scene in scenes],
                                                  aspect_ratio=payload.aspect_ratio, quality=payload.quality,
                                                  audio_mode=payload.audio_mode), workspace_id)
    wall_seconds = sum(scene.wall_seconds for scene in scenes)
    estimated_cost, cost_per_minute, cost_note = estimate_gpu_cost(render_seconds=wall_seconds,
                                                                  gpu=scenes[-1].gpu,
                                                                  output_duration_seconds=payload.duration_seconds * len(scenes))
    return FullVideoGenerationResponse(
        final_video_url=combined.final_video_url, final_download_url=combined.final_download_url,
        final_filename=combined.final_filename, scene_video_urls=[scene.video_url for scene in scenes],
        render_details=[scene.render_details for scene in scenes], total_render_seconds=sum(scene.render_seconds for scene in scenes),
        total_wall_seconds=wall_seconds, provider=scenes[-1].provider, model=payload.model,
        quality=payload.quality, quality_note=combined.quality_note, realism_profile=payload.realism_profile,
        detail_refined=any(scene.detail_refined for scene in scenes), audio_mode=payload.audio_mode,
        gpu=scenes[-1].gpu, media_info=combined.media_info, estimated_cost_usd=estimated_cost,
        estimated_cost_per_output_minute_usd=cost_per_minute, cost_note=cost_note,
        visual_qc_status=combined.visual_qc_status, audio_qc_status=combined.audio_qc_status,
        scene_results=combined.scene_results, warnings=combined.warnings,
    )


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def metrics_summary():
    if settings.is_production and not settings.enable_metrics_endpoint:
        raise HTTPException(status_code=404, detail="Not found.")
    return MetricsSummaryResponse.model_validate(summarize_generation_metrics())


@router.get("/download/{filename}")
async def download_generated_video(filename: str, request: Request):
    try:
        path = resolve_generated_video(filename)
        if settings.auth_enabled or settings.is_production:
            workspace_id = workspace_id_from_request(request)
            if not workspace_id or not workspace_owns_generated_file(workspace_id, path.name):
                raise FileNotFoundError("Generated video not found.")
        return FileResponse(path, media_type="video/mp4", filename=path.name)
    except Exception as exc:
        detail = str(exc) if settings.debug and not settings.is_production else "Generated video not found."
        raise HTTPException(status_code=404, detail=detail) from exc
