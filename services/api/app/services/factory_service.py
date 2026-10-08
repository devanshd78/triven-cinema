import math
import uuid
from pathlib import Path
from typing import Callable

from app.core.config import settings
from app.schemas.factory import FactoryGenerationRequest, FactoryGenerationResponse
from app.schemas.generation import MediaInfo
from app.services.continuity_service import compose_continuity_prompt, compose_render_integrity_prompt
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
from app.services.long_render_service import render_long_clip
from app.services.dialogue_service import build_audio_prompt, separate_visual_and_script
from app.services.job_service import checkpoint_job_result
from app.services.render_review_service import aggregate_qc, pass_flag, qc_report, register_render, review_audio
from app.services.media_probe import probe_media
from app.services.metrics_service import estimate_gpu_cost, record_generation_metric
from app.services.scene_planner import create_prompt_only_plan, create_scene_plan
from app.services.storage_service import ensure_minimum_free_disk
from app.services.video_combiner import combine_videos, extract_continuity_frame
from app.services.video_profiles import source_render_dimensions, validate_factory_scene_duration
from app.services.youtube_service import upload_video
from inference.providers.router import get_video_provider


PROJECT_ROOT = Path(__file__).resolve().parents[4]
GENERATED_DIR = (PROJECT_ROOT / "storage" / "generated").resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
ProgressCallback = Callable[[str, int, str], None]



def _factory_scene_durations(total_seconds: float, max_scene_seconds: float, min_scene_seconds: float) -> list[float]:
    """Split a Factory runtime without creating a short tail scene.

    Example: 30s total with a 20s scene target becomes 15s + 15s, never
    20s + 10s. This keeps every scene at or above the Factory minimum.
    """
    total = float(total_seconds)
    maximum = float(max_scene_seconds)
    minimum = float(min_scene_seconds)
    if total <= 0 or maximum <= 0 or minimum <= 0:
        raise ValueError("Factory durations must be positive.")
    if total + 1e-6 < minimum:
        raise ValueError(f"Factory runtime must be at least {minimum:g}s.")

    count = max(1, math.ceil(total / maximum))
    durations = [maximum] * max(0, count - 1)
    durations.append(total - maximum * max(0, count - 1))

    if len(durations) > 1 and durations[-1] + 1e-6 < minimum:
        deficit = minimum - durations[-1]
        for index in range(len(durations) - 2, -1, -1):
            transferable = max(0.0, durations[index] - minimum)
            moved = min(deficit, transferable)
            durations[index] -= moved
            durations[-1] += moved
            deficit -= moved
            if deficit <= 1e-6:
                break
        if deficit > 1e-6:
            # For supported Factory targets this should not happen; if it does,
            # evenly rebalance rather than silently creating a sub-minimum scene.
            even = total / count
            if even + 1e-6 < minimum:
                raise ValueError("Factory runtime cannot be partitioned into the selected scene profile.")
            durations = [even] * count

    return [round(value, 3) for value in durations]



def _wants_single_continuous_shot(prompt: str) -> bool:
    text = " ".join((prompt or "").lower().split())
    phrases = (
        "one continuous shot",
        "single continuous shot",
        "one continuous static shot",
        "single continuous static shot",
        "filmed as one continuous",
        "without cuts",
        "no cuts",
        "uncut shot",
        "single take",
        "one take",
    )
    return any(phrase in text for phrase in phrases)

def _scene_seed(base_seed: int, scene_index: int, attempt: int = 0) -> int:
    """Return a deterministic but different seed for every scene/retry.

    Reusing the same seed with the same or similar prompt at every 10-second Factory
    boundary strongly biases LTX toward replaying the same composition. First-frame
    conditioning carries identity; the seed should not also freeze scene layout.
    """
    modulus = 2_147_483_648
    return (int(base_seed) + int(scene_index) * 104_729 + int(attempt) * 7_919) % modulus


def _media_info(path: Path) -> MediaInfo:
    return MediaInfo.model_validate(probe_media(path))


def _safe_title(prompt: str) -> str:
    compact = " ".join(prompt.split())
    return (compact[:96].rstrip(" .,:;-") or "Triven Cinema")[:100]


def _qc_unavailable_is_fatal(strict_requested: bool) -> bool:
    """QC is advisory to delivery; a completed video is always retained."""
    return False


def run_factory_generation(
    request: FactoryGenerationRequest,
    *,
    workspace_id: str,
    progress: ProgressCallback | None = None,
) -> FactoryGenerationResponse:
    """Run the prompt-to-publish AI video factory as one bounded background job.

    Strict continuity uses five layers together: immutable character/style bibles,
    explicit entity cardinality, first-frame latent conditioning, stable near-end
    anchor selection, and an optional vision QC/regeneration gate.
    """
    ensure_minimum_free_disk()
    try:
        resolved_element_bindings = resolve_element_bindings(workspace_id, request.element_bindings)
        validate_prompt_bindings(request.prompt, resolved_element_bindings)
    except ElementError as exc:
        raise ValueError(str(exc)) from exc
    if resolved_element_bindings and request.provider != "modal":
        raise ValueError("Reusable Elements currently require the Modal LTX-2.5 provider.")
    if request.realism_profile == "identity_max" and not any(
        item.type == "character" and item.reference_mode == "identity"
        for item in resolved_element_bindings
    ):
        raise ValueError("Identity Max requires an identity-mode Character Element with a real reference image.")

    if request.target_duration_seconds > settings.max_factory_duration_seconds:
        raise ValueError(
            f"Factory jobs are limited to {settings.max_factory_duration_seconds}s on this deployment."
        )
    # Respect explicit single-take intent. A 30s prompt that says "one continuous
    # shot" must not silently become two independently generated 15s clips merely
    # because the generic Factory scene selector was left at 20s. That split is a
    # major source of face/set discontinuity in creator videos.
    effective_scene_seconds = float(request.scene_duration_seconds)
    if (
        request.provider == "modal"
        and request.quality == "1080p"
        and request.target_duration_seconds <= settings.factory_experimental_1080p_scene_seconds + 1e-6
        and request.target_duration_seconds > effective_scene_seconds + 1e-6
        and _wants_single_continuous_shot(request.prompt)
    ):
        effective_scene_seconds = float(request.target_duration_seconds)

    validate_factory_scene_duration(
        quality=request.quality,
        duration_seconds=effective_scene_seconds,
    )
    if request.provider != "modal" and effective_scene_seconds > settings.ltx_native_chunk_seconds:
        raise ValueError("Long factory scenes currently require the Modal LTX provider.")

    scene_durations = _factory_scene_durations(
        request.target_duration_seconds,
        effective_scene_seconds,
        settings.factory_min_scene_seconds,
    )
    scene_count = len(scene_durations)
    visual_prompt, spoken_script = separate_visual_and_script(request.prompt, request.spoken_script)
    planned_scene_seconds = max(scene_durations)
    if scene_count > 20:
        raise ValueError(
            "This factory job would require more than 20 story scenes. Increase scene duration or reduce total duration."
        )

    if progress:
        progress("planning", 8, f"Planning {scene_count} identity/cardinality-locked scenes...")
    if request.enhance_prompt:
        plan = create_scene_plan(
            prompt=visual_prompt,
            spoken_script=spoken_script,
            scene_count=scene_count,
            aspect_ratio=request.aspect_ratio,
            force_ai=True,
            target_scene_duration_seconds=planned_scene_seconds,
        )
    else:
        plan = create_prompt_only_plan(
            prompt=visual_prompt,
            spoken_script=spoken_script,
            scene_count=scene_count,
            target_scene_duration_seconds=planned_scene_seconds,
        )

    if (
        request.enhance_prompt
        and request.quality != "preview"
        and plan.source == "fallback"
        and not settings.factory_allow_fallback_final
    ):
        raise RuntimeError(
            "Final-quality Factory planning could not use Gemini after retries/failover. "
            "The local fallback is intentionally blocked before GPU rendering because it "
            "cannot safely guarantee named-character continuity for this job. Retry when "
            "Gemini is available, or use Preview if you explicitly want fallback planning."
        )

    provider = get_video_provider(request.provider, model=request.model)
    render_mode = "dfr" if request.quality != "preview" and request.provider == "modal" else "distilled"
    effective_decoder = "diffusion" if render_mode == "dfr" else request.decoder
    width, height = source_render_dimensions(request.aspect_ratio, request.quality)
    continuity_id = f"factory-{uuid.uuid4().hex[:16]}"
    source_paths: list[Path] = []
    render_details: list[str] = []
    total_render = 0.0
    total_wall = 0.0
    total_chunks = 0
    any_detail_refined = False
    gpu: str | None = None
    previous_frame: Path | None = None
    continuity_warnings: list[str] = []
    continuity_regenerations = 0
    audio_retake_count = 0
    audio_warnings: list[str] = []
    elements_used: set[str] = set()
    element_reference_modes: set[str] = set()
    temporary_element_sheets: list[Path] = []
    scene_results: list[dict] = []
    delivery_warnings: list[str] = []

    try:
        for index, scene in enumerate(plan.scenes):
            if index >= len(scene_durations):
                break
            duration = scene_durations[index]
            scene_script = getattr(scene, "spoken_script", "") or (spoken_script if scene_count == 1 else "")
            scene_attempts: list[dict] = []
            visual_report = qc_report()
            active_elements = elements_for_scene(scene.prompt, resolved_element_bindings)
            for binding in active_elements:
                elements_used.add(f"@{binding.handle}")

            identity_elements = [item for item in active_elements if item.reference_mode == "identity"]
            start_frame_elements = [item for item in active_elements if item.reference_mode == "start_frame"]
            character_identity_elements = [item for item in identity_elements if item.type == "character"]
            prompt_wardrobe_authoritative = any(
                item.wardrobe_policy == "prompt" for item in character_identity_elements
            )
            use_ingredients = bool(identity_elements or len(active_elements) > 1)
            if use_ingredients and not settings.element_ingredients_enabled:
                raise ValueError("Element identity conditioning is disabled on this deployment.")
            if use_ingredients and duration > settings.element_ingredients_max_scene_seconds + 1e-6:
                raise ValueError(
                    f"Element identity-conditioned Factory scenes are currently limited to "
                    f"{settings.element_ingredients_max_scene_seconds:g}s on this deployment."
                )

            element_sheet_path: Path | None = None
            if use_ingredients:
                element_sheet_path = GENERATED_DIR / f".element-sheet-{uuid.uuid4().hex}.png"
                build_reference_sheet(active_elements, element_sheet_path)
                temporary_element_sheets.append(element_sheet_path)
                element_reference_modes.add("ingredients")

            explicit_start_frame = Path(start_frame_elements[0].primary_asset_path) if start_frame_elements else None
            scene_reference_frame = explicit_start_frame or (
                previous_frame if request.continuity_mode == "strict" and previous_frame is not None else None
            )
            if explicit_start_frame is not None:
                element_reference_modes.add("start_frame")

            base_progress = 12 + int((index / max(1, scene_count)) * 66)

            def chunk_progress(part: int, count: int, message: str) -> None:
                if not progress:
                    return
                within_scene = int(((part + 1) / max(1, count)) * max(1, 66 // max(1, scene_count)))
                progress(
                    "rendering",
                    min(78, base_progress + within_scene),
                    f"Scene {index + 1}/{scene_count} · {message}",
                )

            accepted_result = None
            accepted_path: Path | None = None
            last_qc_note = ""
            attempts = max(0, int(request.continuity_max_retries)) + 1
            # Prompt-only Factory mode still receives strict *inspection*. QC never
            # rewrites the user's creative prompt; it only rejects/regenerates broken
            # identity, wardrobe or artifact frames.
            effective_qc_mode = request.continuity_qc_mode

            for attempt in range(attempts):
                locked_prompt = scene.prompt
                if request.enhance_prompt and request.continuity_mode != "off":
                    locked_prompt = compose_continuity_prompt(
                        scene_prompt=locked_prompt,
                        character_bible=plan.character_bible,
                        style_bible=plan.style_bible,
                        scene_index=index,
                        scene_count=scene_count,
                        entity_locks=plan.entity_locks,
                        visible_entity_counts=scene.visible_entity_counts,
                        reference_frame_present=(scene_reference_frame is not None),
                        prompt_wardrobe_authoritative=prompt_wardrobe_authoritative,
                        retry_level=attempt,
                        qc_feedback=last_qc_note,
                    )
                locked_prompt = compose_render_integrity_prompt(
                    locked_prompt,
                    realism_profile=request.realism_profile,
                    prompt_wardrobe_authoritative=prompt_wardrobe_authoritative,
                    retry_level=attempt,
                    qc_feedback=last_qc_note,
                )
                if active_elements:
                    locked_prompt = compile_element_prompt(locked_prompt, active_elements)
                locked_prompt = build_audio_prompt(locked_prompt, scene_script, request.audio_direction)

                if attempt > 0 and progress:
                    progress(
                        "rendering",
                        min(77, base_progress + 2),
                        f"Scene {index + 1}/{scene_count} · continuity QC retry {attempt}/{attempts - 1}",
                    )

                try:
                    result = render_long_clip(
                        provider=provider,
                        prompt=locked_prompt,
                        width=width,
                        height=height,
                        duration_seconds=duration,
                        seed=_scene_seed(request.seed, index, attempt),
                        decoder=effective_decoder,
                        enhance_prompt=False,
                        render_mode=render_mode,
                        reference_image_path=str(scene_reference_frame) if scene_reference_frame is not None else None,
                        reference_strength=(
                            1.0
                            if request.realism_profile == "identity_max" and character_identity_elements
                            else max(request.continuity_strength, 0.95 if character_identity_elements and request.continuity_mode == "strict" else request.continuity_strength)
                        ),
                        element_reference_sheet_path=str(element_sheet_path) if element_sheet_path is not None else None,
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
                    if accepted_path is None:
                        raise
                    continuity_warnings.append(
                        f"Scene {index + 1}: regeneration failed ({type(exc).__name__}); keeping the previous rendered video for review."
                    )
                    break
                path = Path(result.path)
                register_render(workspace_id, path, kind="raw", scene_index=index, attempt=attempt,
                                visual_qc_status="not_checked", audio_qc_status="not_checked")
                accepted_result, accepted_path = result, path

                total_render += float(result.render_seconds)
                total_wall += float(result.wall_seconds or result.render_seconds)
                total_chunks += int(result.chunk_count or 1)
                gpu = result.gpu or gpu

                qc = None
                if effective_qc_mode != "off":
                    if progress:
                        progress("rendering", min(79, base_progress + 2), f"Scene {index + 1}/{scene_count} · inspecting visual identity and artifacts...")
                    try:
                        qc = evaluate_scene_cardinality(
                            path, entity_locks=plan.entity_locks,
                            visible_entity_counts=scene.visible_entity_counts,
                            character_bible=plan.character_bible, scene_prompt=locked_prompt,
                            qc_mode=effective_qc_mode, reference_frame_path=scene_reference_frame,
                            canonical_reference_paths=element_canonical_reference_paths(active_elements),
                        )
                    except Exception as exc:
                        qc = ContinuityQCResult(skipped=True, note=f"Visual QC unavailable ({type(exc).__name__}).")
                visual_report = qc_report(qc)
                scene_attempts.append({"filename": path.name, "attempt": attempt + 1, **visual_report})
                register_render(workspace_id, path, kind="scene", scene_index=index,
                                visual_qc_status=visual_report["status"], visual_qc=visual_report,
                                visual_qc_attempts=scene_attempts)
                if visual_report["status"] != "failed":
                    if visual_report["status"] == "unavailable":
                        continuity_warnings.append(f"Scene {index + 1}: {visual_report['note']}")
                    break
                last_qc_note = visual_report.get("note") or "; ".join(visual_report.get("violations") or []) or "Visual quality failure"
                if attempt < attempts - 1:
                    continuity_regenerations += 1
                    # Keep this owned candidate until and after a replacement succeeds.
                    continue
                continuity_warnings.append(
                    f"Scene {index + 1} continuity QC failed after {attempt + 1} attempt(s): {last_qc_note}. Video retained for review; visual QC did not pass."
                )

            if accepted_result is None or accepted_path is None:
                raise RuntimeError(f"Scene {index + 1} did not produce a render.")
            any_detail_refined = any_detail_refined or bool(accepted_result.detail_refined)
            if visual_report["status"] == "failed" and not any(f"Scene {index + 1} continuity QC failed" in item for item in continuity_warnings):
                continuity_warnings.append(f"Scene {index + 1}: {last_qc_note}. Video retained for review; visual QC did not pass.")
            audio_review = review_audio(
                accepted_path, provider=provider, workspace_id=workspace_id,
                scene_prompt=scene.prompt, spoken_script=scene_script,
                audio_direction=request.audio_direction, audio_mode=request.audio_mode,
                duration_seconds=duration, seed=_scene_seed(request.seed, index), scene_index=index,
            )
            accepted_path = audio_review.path
            audio_warnings.extend(audio_review.warnings)
            if audio_review.retake is not None:
                retake = audio_review.retake
                audio_retake_count += 1
                total_render += float(retake.render_seconds)
                total_wall += float(retake.wall_seconds or retake.render_seconds)
                total_chunks += int(retake.chunk_count or 1)
                gpu = retake.gpu or gpu
                render_details.append(retake.render_details)
            scene_record = {
                "scene_index": index, "filename": accepted_path.name,
                "video_url": f"/media/generated/{accepted_path.name}",
                "download_url": f"/api/v1/generations/download/{accepted_path.name}",
                "spoken_script": scene_script,
                "visual_qc_status": visual_report["status"], "visual_qc": visual_report,
                "audio_qc_status": audio_review.report["status"], "audio_qc": audio_review.report,
                "visual_qc_attempts": scene_attempts, "audio_qc_attempts": audio_review.attempts,
            }
            scene_results.append(scene_record)
            register_render(workspace_id, accepted_path, **scene_record)
            checkpoint_job_result({"scene_results": scene_results,
                                   "visual_qc_status": aggregate_qc([x["visual_qc"] for x in scene_results]),
                                   "audio_qc_status": aggregate_qc([x["audio_qc"] for x in scene_results]),
                                   "continuity_warnings": continuity_warnings, "audio_warnings": audio_warnings})

            source_paths.append(accepted_path)
            render_details.append(accepted_result.render_details)

            if visual_report["status"] == "failed":
                previous_frame = None
            if request.continuity_mode != "off" and visual_report["status"] != "failed" and index < scene_count - 1:
                frame = GENERATED_DIR / f".factory-continuity-{uuid.uuid4().hex}.png"
                try:
                    extract_continuity_frame(accepted_path, frame)
                    register_render(workspace_id, frame, kind="continuity_frame", scene_index=index)
                    previous_frame = frame
                except Exception as exc:
                    previous_frame = None
                    continuity_warnings.append(f"Scene {index + 1}: continuity anchor unavailable ({type(exc).__name__}); original references remain active.")

        if not source_paths:
            raise RuntimeError("Factory produced no scene clips.")

        if progress:
            progress("composing", 80, "Composing accepted scenes and synchronized audio...")
        composed = GENERATED_DIR / f"factory-source-{uuid.uuid4().hex}.mp4"
        delivery_complete = True
        try:
            combine_videos(source_paths, composed)
            register_render(workspace_id, composed, kind="composed")
        except Exception as exc:
            composed = source_paths[0]
            delivery_complete = len(source_paths) == 1
            delivery_warnings.append(f"Composition unavailable ({type(exc).__name__}); rendered scenes remain available individually.")
        if progress:
            progress("delivery", 88, f"Creating {request.quality} delivery master...")
        try:
            delivery = prepare_delivery(composed, aspect_ratio=request.aspect_ratio, quality=request.quality,
                                        audio_mode=request.audio_mode, prefix="factory")
            register_render(workspace_id, delivery, kind="delivery")
        except Exception as exc:
            delivery = composed
            delivery_complete = False
            delivery_warnings.append(f"Delivery processing unavailable ({type(exc).__name__}); original rendered video retained.")
        try:
            info = _media_info(delivery)
        except Exception as exc:
            info = MediaInfo()
            delivery_complete = False
            delivery_warnings.append(f"Media inspection unavailable ({type(exc).__name__}); rendered file retained.")
        checkpoint_job_result({"final_filename": delivery.name, "final_video_url": f"/media/generated/{delivery.name}",
                               "final_download_url": f"/api/v1/generations/download/{delivery.name}",
                               "delivery_complete": delivery_complete, "warnings": delivery_warnings})

        estimated_cost, cost_per_minute, cost_note = estimate_gpu_cost(
            render_seconds=total_wall,
            gpu=gpu,
            output_duration_seconds=request.target_duration_seconds,
        )

        visual_qc_status = aggregate_qc([item["visual_qc"] for item in scene_results])
        audio_qc_status = aggregate_qc([item["audio_qc"] for item in scene_results])
        register_render(workspace_id, delivery, kind="delivery", scene_results=scene_results,
                        visual_qc_status=visual_qc_status, audio_qc_status=audio_qc_status,
                        delivery_complete=delivery_complete, quality=request.quality)
        youtube_video_id: str | None = None
        youtube_url: str | None = None
        youtube_privacy: str | None = None
        # Never automatically publish a QC-rejected video. The master remains
        # available for download and explicit human review instead.
        unapproved_qc = visual_qc_status != "passed" or (request.audio_mode != "mute" and audio_qc_status != "passed") or not delivery_complete
        if request.publish_to_youtube and unapproved_qc:
            continuity_warnings.append(
                "Automatic YouTube publishing skipped because visual or audio QC did not pass or could not complete. Review the saved video first."
            )
        if request.publish_to_youtube and not unapproved_qc:
            if progress:
                progress("publishing", 95, "Uploading the finished master to the connected YouTube channel...")
            try:
                published = upload_video(
                    workspace_id,
                    filename=delivery.name,
                    title=(request.youtube_title or _safe_title(request.prompt)),
                    description=request.youtube_description,
                    privacy=request.youtube_privacy,
                    tags=request.youtube_tags,
                    category_id=request.youtube_category_id,
                    publish_at=request.youtube_publish_at,
                )
                youtube_video_id = published["video_id"]
                youtube_url = published["youtube_url"]
                youtube_privacy = published["privacy"]
            except Exception as exc:
                delivery_warnings.append(f"YouTube publication failed ({type(exc).__name__}); the completed video remains available for download.")

        filename = delivery.name
        qc_passed = pass_flag(visual_qc_status)

        try:
            record_generation_metric(
                {
                    "type": "factory",
                    "provider": provider.name,
                    "model": request.model,
                    "gpu": gpu,
                    "aspect_ratio": request.aspect_ratio,
                    "scene_count": len(source_paths),
                    "chunk_count": total_chunks,
                    "target_duration_seconds": request.target_duration_seconds,
                    "scene_duration_seconds": effective_scene_seconds,
                    "render_seconds": round(total_render, 3),
                    "wall_seconds": round(total_wall, 3),
                    "quality": request.quality,
                    "audio_mode": request.audio_mode,
                    "has_audio": info.has_audio,
                    "entity_lock_count": len(plan.entity_locks),
                    "continuity_qc_mode": request.continuity_qc_mode,
                    "continuity_qc_passed": qc_passed,
                    "visual_qc_status": visual_qc_status,
                    "continuity_regenerations": continuity_regenerations,
                    "continuity_warning_count": len(continuity_warnings),
                    "audio_qc_passed": pass_flag(audio_qc_status),
                    "audio_qc_status": audio_qc_status,
                    "audio_retake_count": audio_retake_count,
                    "audio_warning_count": len(audio_warnings),
                    "render_mode": render_mode,
                    "realism_profile": request.realism_profile,
                    "detail_refined": any_detail_refined,
                    "estimated_cost_usd": estimated_cost,
                    "estimated_cost_per_output_minute_usd": cost_per_minute,
                    "youtube_published": bool(youtube_url),
                    "filename": filename,
                }
            )
        except Exception:
            delivery_warnings.append("Usage metrics could not be recorded; the rendered video is available.")

        response = FactoryGenerationResponse(
            scene_results=scene_results,
            warnings=delivery_warnings,
            delivery_complete=delivery_complete,
            final_video_url=f"/media/generated/{filename}",
            final_download_url=f"/api/v1/generations/download/{filename}",
            final_filename=filename,
            target_duration_seconds=request.target_duration_seconds,
            actual_duration_seconds=info.duration_seconds,
            scene_count=len(source_paths),
            scene_duration_seconds=planned_scene_seconds,
            aspect_ratio=request.aspect_ratio,
            quality=request.quality,
            quality_note=(quality_note(request.quality, request.aspect_ratio) if delivery_complete else
                          "Original rendered source retained; requested delivery processing did not complete."),
            realism_profile=request.realism_profile,
            detail_refined=any_detail_refined,
            audio_mode=request.audio_mode,
            has_audio=info.has_audio,
            width=info.width,
            height=info.height,
            provider=provider.name,
            model=request.model,
            gpu=gpu,
            total_render_seconds=round(total_render, 2),
            total_wall_seconds=round(total_wall, 2),
            chunk_count=total_chunks,
            estimated_cost_usd=estimated_cost,
            estimated_cost_per_output_minute_usd=cost_per_minute,
            cost_note=cost_note,
            planner_source=plan.source,
            planner_note=plan.note,
            continuity_id=continuity_id,
            entity_locks=plan.entity_locks,
            continuity_qc_passed=qc_passed,
            visual_qc_status=visual_qc_status,
            continuity_regenerations=continuity_regenerations,
            continuity_warnings=continuity_warnings,
            audio_qc_passed=pass_flag(audio_qc_status),
            audio_qc_status=audio_qc_status,
            audio_retake_count=audio_retake_count,
            audio_warnings=audio_warnings,
            elements_used=sorted(elements_used),
            element_reference_mode=("+".join(sorted(element_reference_modes)) if element_reference_modes else None),
            youtube_video_id=youtube_video_id,
            youtube_url=youtube_url,
            youtube_privacy=youtube_privacy,
        )
        checkpoint_job_result(response.model_dump())
        return response
    finally:
        for sheet in temporary_element_sheets:
            sheet.unlink(missing_ok=True)
