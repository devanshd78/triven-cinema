from pathlib import Path


MODEL_ROOT = Path("/models/ltx-2.5")

TRANSFORMER = MODEL_ROOT / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER = MODEL_ROOT / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE_DIFFUSION = MODEL_ROOT / "vae/ltx-2.5-video-vae-bf16.safetensors"
VIDEO_VAE_CONV = MODEL_ROOT / "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO_VAE = MODEL_ROOT / "vae/ltx-2.5-audio-vae-bf16.safetensors"
SPATIAL_UPSCALER = MODEL_ROOT / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
DETAILING_LORA = MODEL_ROOT / "loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
INGREDIENTS_LORA = MODEL_ROOT / "loras/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors"
REFINE_DETAILS_LORA = MODEL_ROOT / "loras/ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors"

BASE_MODEL_FILES = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]

REQUIRED_MODEL_FILES = BASE_MODEL_FILES
DFR_REQUIRED_FILES = [DETAILING_LORA]
INGREDIENTS_REQUIRED_FILES = [INGREDIENTS_LORA]
REFINE_DETAILS_REQUIRED_FILES = [REFINE_DETAILS_LORA]


def required_model_paths(*, render_mode="distilled", realism_profile="standard", element_reference_required=False, operation="generate", decoder="conv") -> list[Path]:
    """Dependencies of the selected recipe, rather than every optional adapter."""
    if operation not in {"generate", "retake_audio"}:
        raise ValueError(f"Unsupported inference operation: {operation}")
    if render_mode not in {"distilled", "dfr"}:
        raise ValueError(f"Unsupported render mode: {render_mode}")
    if realism_profile not in {"standard", "real_skin", "identity_max"}:
        raise ValueError(f"Unsupported realism profile: {realism_profile}")
    if decoder not in {"conv", "diffusion"}:
        raise ValueError(f"Unsupported decoder: {decoder}")
    diffusion = operation == "retake_audio" or decoder == "diffusion" or (render_mode == "dfr" and not element_reference_required)
    paths = [TRANSFORMER, TEXT_ENCODER, AUDIO_VAE, VIDEO_VAE_DIFFUSION if diffusion else VIDEO_VAE_CONV]
    if operation == "generate":
        paths.append(SPATIAL_UPSCALER)
        if element_reference_required:
            paths.append(INGREDIENTS_LORA)
        elif render_mode == "dfr":
            paths.append(DETAILING_LORA)
        if render_mode == "dfr" and realism_profile in {"real_skin", "identity_max"}:
            paths.extend([REFINE_DETAILS_LORA, VIDEO_VAE_DIFFUSION])
    return list(dict.fromkeys(paths))


def model_hub_reference(path: Path) -> tuple[str, str]:
    adapters = {
        DETAILING_LORA: "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler",
        INGREDIENTS_LORA: "Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients",
        REFINE_DETAILS_LORA: "Lightricks/LTX-2.5-22b-IC-LoRA-Refine-Details",
    }
    if path in adapters:
        return adapters[path], path.name
    return "Lightricks/LTX-2.5", path.relative_to(MODEL_ROOT).as_posix()


def check_model_file_access(paths: list[Path], token: str) -> list[str]:
    """Authenticated HEADs verify file access, including gated weight permission.

    Repository metadata can be public while its weight downloads are gated.
    This check downloads no model bytes and also detects truncated local files
    when Hugging Face supplies the expected content length.
    """
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    def inspect(path: Path) -> str | None:
        repo, filename = model_hub_reference(path)
        try:
            metadata = get_hf_file_metadata(hf_hub_url(repo, filename), token=token, timeout=10)
            if metadata.size is not None and path.is_file() and path.stat().st_size != metadata.size:
                return f"Model file size mismatch for {path.relative_to(MODEL_ROOT)}; rerun `modal run modal/app.py::download_models`."
            return None
        except Exception as exc:
            return (f"Hugging Face weight access failed for {repo}/{filename} ({type(exc).__name__}); "
                    "verify gated-model approval, HF_TOKEN permissions and network connectivity.")

    with ThreadPoolExecutor(max_workers=4) as pool:
        return [error for error in pool.map(inspect, paths) if error]
