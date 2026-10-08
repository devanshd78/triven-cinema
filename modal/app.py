import os
import re
import subprocess
import time
import uuid
from pathlib import Path

import modal

from ltx_worker import (
    build_command,
    build_refine_details_command,
    ingredients_temporal_chunk_count,
    make_static_reference_video,
    preserve_source_audio,
    run_ltx_command,
    static_reference_frame_count,
    temporal_chunk_count,
)
from models import DETAILING_LORA, INGREDIENTS_LORA, MODEL_ROOT, REFINE_DETAILS_LORA, REQUIRED_MODEL_FILES, required_model_paths, check_model_file_access
from retake_worker import retake_audio_only


APP_NAME = "triven-cinema-ltx"
GPU_TYPE = os.getenv("TRIVEN_MODAL_GPU", "B200")
SCALEDOWN_WINDOW = int(os.getenv("TRIVEN_MODAL_SCALEDOWN_WINDOW", "15"))
# Pin a tested public LTX release. Do not silently build production workers from
# a moving `main` branch; upgrade this only after preview + regression validation.
LTX_REPO_REF = os.getenv("TRIVEN_LTX_REPO_REF", "v1.4.2").strip() or "v1.4.2"
if not re.fullmatch(r"[A-Za-z0-9._/-]+", LTX_REPO_REF):
    raise RuntimeError("TRIVEN_LTX_REPO_REF contains unsupported characters.")

app = modal.App(APP_NAME)

models_volume = modal.Volume.from_name(
    "triven-cinema-models",
    create_if_missing=True,
)

hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "build-essential")
    .uv_pip_install("uv", "huggingface_hub")
    .run_commands(
        "git clone --depth 1 https://github.com/Lightricks/LTX-2.git /opt/LTX-2",
        f"cd /opt/LTX-2 && git fetch --depth 1 origin {LTX_REPO_REF} && git checkout --detach FETCH_HEAD",
        "cd /opt/LTX-2 && uv sync --extra natten",
    )
    .add_local_python_source("ltx_worker", "models", "retake_worker")
)


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={"/models": models_volume},
    timeout=60 * 60,
)
def download_models() -> dict:
    """Download LTX-2.5 base weights plus production/detail IC-LoRAs once."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from the Modal secret `huggingface-secret`.")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename in REQUIRED_MODEL_FILES:
        path = hf_hub_download(
            repo_id="Lightricks/LTX-2.5",
            filename=filename,
            local_dir=str(MODEL_ROOT),
            token=token,
        )
        downloaded.append(path)

    DETAILING_LORA.parent.mkdir(parents=True, exist_ok=True)
    detailing = hf_hub_download(
        repo_id="Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
        filename=DETAILING_LORA.name,
        local_dir=str(DETAILING_LORA.parent),
        token=token,
    )
    downloaded.append(detailing)

    INGREDIENTS_LORA.parent.mkdir(parents=True, exist_ok=True)
    ingredients = hf_hub_download(
        repo_id="Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
        filename=INGREDIENTS_LORA.name,
        local_dir=str(INGREDIENTS_LORA.parent),
        token=token,
    )
    downloaded.append(ingredients)

    REFINE_DETAILS_LORA.parent.mkdir(parents=True, exist_ok=True)
    refine_details = hf_hub_download(
        repo_id="Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details",
        filename=REFINE_DETAILS_LORA.name,
        local_dir=str(REFINE_DETAILS_LORA.parent),
        token=token,
    )
    downloaded.append(refine_details)

    models_volume.commit()
    return {"downloaded": downloaded, "count": len(downloaded), "ltx_repo_ref": LTX_REPO_REF}


def _missing_base_models() -> list[str]:
    return [path for path in REQUIRED_MODEL_FILES if not (MODEL_ROOT / path).exists()]


@app.function(image=image, secrets=[hf_secret], volumes={"/models": models_volume}, timeout=120)
def preflight(render_mode: str = "distilled", realism_profile: str = "standard", element_reference_required: bool = False, operation: str = "generate", decoder: str = "conv") -> dict:
    """CPU-only manifest check; never allocate a GPU to discover missing weights."""
    models_volume.reload()
    paths = required_model_paths(render_mode=render_mode, realism_profile=realism_profile,
                                 element_reference_required=element_reference_required, operation=operation, decoder=decoder)
    errors = []
    missing = [str(path.relative_to(MODEL_ROOT)) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        errors.append("Missing or empty model files: " + ", ".join(missing) + ". Run `modal run modal/app.py::download_models`.")
    interpreter = Path("/opt/LTX-2/.venv/bin/python")
    if not interpreter.is_file():
        errors.append("LTX Python environment is missing; redeploy `modal deploy modal/app.py`.")
    else:
        check = subprocess.run([str(interpreter), "-c", "import importlib.util; assert all(importlib.util.find_spec(m) for m in ['torch','ltx_pipelines.ic_lora','ltx_pipelines.retake','ltx_pipelines.dfr_pipeline']), 'Required LTX modules missing'"], capture_output=True, text=True, timeout=45)
        if check.returncode:
            errors.append("Worker package compatibility check failed; redeploy the pinned LTX image. " + check.stderr[-500:])
    if not GPU_TYPE.strip():
        errors.append("TRIVEN_MODAL_GPU is empty; configure a Modal GPU and redeploy.")
    # Inference can use cached weights, but validate model access before accepting a
    # deployment as ready so recovery/downloads do not fail after a job is queued.
    token = os.environ.get("HF_TOKEN")
    if not token:
        errors.append("HF_TOKEN is missing from the Modal secret `huggingface-secret`.")
    else:
        errors.extend(check_model_file_access(paths, token))
    return {"ready": not errors, "supported": True, "protocol_version": 2, "ltx_repo_ref": LTX_REPO_REF,
            "gpu": GPU_TYPE, "operation": operation, "required_models": [str(p.relative_to(MODEL_ROOT)) for p in paths],
            "missing_models": missing, "errors": errors}


@app.function(
    image=image,
    gpu=GPU_TYPE,
    secrets=[hf_secret],
    volumes={"/models": models_volume},
    timeout=60 * 60,
    scaledown_window=SCALEDOWN_WINDOW,
)
def generate_video(
    prompt: str,
    width: int = 1024,
    height: int = 576,
    duration_seconds: float = 1.0,
    seed: int = 42,
    decoder: str = "conv",
    enhance_prompt: bool = False,
    render_mode: str = "distilled",
    reference_image_bytes: bytes | None = None,
    reference_image_suffix: str = ".png",
    reference_strength: float = 0.95,
    element_reference_sheet_bytes: bytes | None = None,
    element_reference_strength: float = 1.0,
    realism_profile: str = "standard",
) -> dict:
    del enhance_prompt  # Prompt enhancement currently happens in Triven/Gemini.

    mode = (render_mode or "distilled").strip().lower()
    realism = (realism_profile or "standard").strip().lower()
    if realism not in {"standard", "real_skin", "identity_max"}:
        raise ValueError(f"Unsupported realism profile: {realism_profile}")
    apply_detail_refiner = mode == "dfr" and realism in {"real_skin", "identity_max"}
    missing = [str(path.relative_to(MODEL_ROOT)) for path in required_model_paths(render_mode=mode, realism_profile=realism, element_reference_required=bool(element_reference_sheet_bytes), decoder=decoder) if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            "LTX-2.5 model files are missing. Run `modal run modal/app.py::download_models` "
            f"before generation. Missing: {missing[:3]}"
        )

    started = time.perf_counter()
    output_path = Path("/tmp") / f"ltx-{uuid.uuid4().hex}.mp4"
    base_output_path = output_path if not apply_detail_refiner else Path("/tmp") / f"ltx-base-{uuid.uuid4().hex}.mp4"
    refined_picture_path = None if not apply_detail_refiner else Path("/tmp") / f"ltx-refined-{uuid.uuid4().hex}.mp4"
    detail_refined = False
    refinement_warning = ""

    reference_path: Path | None = None
    element_sheet_path: Path | None = None
    element_reference_video_path: Path | None = None
    if reference_image_bytes:
        suffix = reference_image_suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            suffix = ".png"
        reference_path = Path("/tmp") / f"continuity-{uuid.uuid4().hex}{suffix}"
        reference_path.write_bytes(reference_image_bytes)

    if element_reference_sheet_bytes:
        element_sheet_path = Path("/tmp") / f"element-sheet-{uuid.uuid4().hex}.png"
        element_sheet_path.write_bytes(element_reference_sheet_bytes)
        element_reference_video_path = Path("/tmp") / f"element-guide-{uuid.uuid4().hex}.mp4"
        make_static_reference_video(
            element_sheet_path,
            element_reference_video_path,
            width=width,
            height=height,
            frame_count=static_reference_frame_count(duration_seconds),
        )

    try:
        command = build_command(
            prompt=prompt,
            output_path=base_output_path,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            seed=seed,
            decoder=decoder,
            render_mode=mode,
            reference_image_path=reference_path,
            reference_strength=reference_strength,
            element_reference_video_path=element_reference_video_path,
            element_reference_strength=element_reference_strength,
        )
        run_ltx_command(command)
        if apply_detail_refiner:
            assert refined_picture_path is not None
            try:
                refine_command = build_refine_details_command(
                    input_video_path=base_output_path,
                    output_path=refined_picture_path,
                    width=width,
                    height=height,
                    duration_seconds=duration_seconds,
                    seed=seed + 97_531,
                )
                run_ltx_command(refine_command)
                preserve_source_audio(
                    refined_video_path=refined_picture_path,
                    source_video_path=base_output_path,
                    output_path=output_path,
                )
                detail_refined = True
            except Exception as exc:
                # Refinement is optional: a successful base render is a deliverable.
                # Return the existing file directly, without a second large copy
                # that could fail when disk space was the refinement failure.
                output_path.unlink(missing_ok=True)
                output_path = base_output_path
                refinement_warning = f"Refinement unavailable; base render preserved ({type(exc).__name__}: {str(exc)[:300]})"
    finally:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
        if element_sheet_path is not None:
            element_sheet_path.unlink(missing_ok=True)
        if element_reference_video_path is not None:
            element_reference_video_path.unlink(missing_ok=True)
        if apply_detail_refiner and output_path != base_output_path and output_path.exists() and output_path.stat().st_size:
            base_output_path.unlink(missing_ok=True)
        if refined_picture_path is not None:
            refined_picture_path.unlink(missing_ok=True)

    if not output_path.exists():
        raise RuntimeError("LTX finished without producing an MP4 file.")

    elapsed = time.perf_counter() - started
    video_bytes = output_path.read_bytes()
    output_path.unlink(missing_ok=True)

    return {
        "video_bytes": video_bytes,
        "seed": seed,
        "prompt": prompt,
        "render_details": (
            f"{width}x{height} · {duration_seconds:.2f}s · "
            + (
                f"Elements Ingredients IC-LoRA · stage-2 identity lock · {static_reference_frame_count(duration_seconds)}f matched guide"
                if element_reference_sheet_bytes
                else ("DFR production · single-pass scene" if mode == "dfr" else "Distilled preview")
            )
            + f" · {('diffusion' if mode == 'dfr' else decoder)} decoder · {GPU_TYPE}"
            + (
                " · Refine Details IC-LoRA · tiled texture pass · source audio preserved"
                if detail_refined
                else ""
            )
            + (" · Identity Max" if realism == "identity_max" else (" · Real Skin" if realism == "real_skin" else ""))
            + (" · first-frame continuity" if reference_image_bytes else "")
            + (f" · WARNING: {refinement_warning}" if refinement_warning else "")
        ),
        "render_seconds": elapsed,
        "gpu": GPU_TYPE,
        "reference_conditioned": bool(reference_image_bytes or element_reference_sheet_bytes),
        "chunk_count": (
            ingredients_temporal_chunk_count(duration_seconds)
            if element_reference_sheet_bytes
            else (temporal_chunk_count(duration_seconds) if mode == "distilled" else 1)
        ),
        "render_mode": "ingredients" if element_reference_sheet_bytes else mode,
        "realism_profile": realism,
        "detail_refined": detail_refined,
        "warnings": [refinement_warning] if refinement_warning else [],
    }


@app.function(
    image=image,
    gpu=GPU_TYPE,
    secrets=[hf_secret],
    volumes={"/models": models_volume},
    timeout=60 * 60,
    scaledown_window=SCALEDOWN_WINDOW,
)
def retake_audio(
    video_bytes: bytes,
    prompt: str,
    duration_seconds: float,
    seed: int = 42,
) -> dict:
    """Regenerate only the LTX audio stream while freezing the source video."""
    missing = [str(path.relative_to(MODEL_ROOT)) for path in required_model_paths(operation="retake_audio")
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            "LTX-2.5 model files are missing. Run `modal run modal/app.py::download_models`. "
            f"Missing: {missing[:3]}"
        )
    if not video_bytes:
        raise ValueError("retake_audio requires source video bytes.")

    started = time.perf_counter()
    input_path = Path("/tmp") / f"retake-input-{uuid.uuid4().hex}.mp4"
    output_path = Path("/tmp") / f"retake-output-{uuid.uuid4().hex}.mp4"
    input_path.write_bytes(video_bytes)
    try:
        retake_audio_only(
            input_path=input_path,
            output_path=output_path,
            prompt=prompt,
            start_time=0.0,
            end_time=max(0.05, float(duration_seconds)),
            seed=seed,
        )
        if not output_path.exists():
            raise RuntimeError("LTX audio Retake finished without producing an MP4 file.")
        result_bytes = output_path.read_bytes()
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    return {
        "video_bytes": result_bytes,
        "seed": seed,
        "prompt": prompt,
        "render_details": f"LTX Retake · audio-only · video preserved · {GPU_TYPE}",
        "render_seconds": time.perf_counter() - started,
        "gpu": GPU_TYPE,
        "reference_conditioned": True,
        "chunk_count": 1,
        "render_mode": "audio-retake",
    }


@app.local_entrypoint()
def main():
    print(f"Triven Cinema Modal app: {APP_NAME}")
    print(f"GPU: {GPU_TYPE}")
    print(f"Scaledown window: {SCALEDOWN_WINDOW}s")
    print(f"LTX repo ref: {LTX_REPO_REF}")
    print("Prepare models: modal run modal/app.py::download_models")
    print("Deploy: modal deploy modal/app.py")
