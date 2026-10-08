import uuid
from pathlib import Path

from app.services.video_combiner import process_audio, transcode_delivery
from app.services.job_service import register_current_generated_asset


GENERATED_DIR = (Path(__file__).resolve().parents[4] / "storage" / "generated").resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def quality_note(quality: str, aspect_ratio: str) -> str:
    if quality == "preview":
        return "LTX source render profile; no delivery upscale applied."

    if quality == "4k":
        if aspect_ratio == "9:16":
            delivery = "2160x3840"
        elif aspect_ratio == "1:1":
            delivery = "2160x2160"
        else:
            delivery = "3840x2160"
        return (
            f"{delivery} 4K delivery master from the LTX-2.5 DFR production path. "
            "Triven targets the LTX 4K grid (3840x2176 landscape / rotated portrait) "
            "and performs one final delivery crop/transcode to the requested aspect."
        )

    if aspect_ratio == "9:16":
        delivery = "1080x1920"
    elif aspect_ratio == "1:1":
        delivery = "1080x1080"
    else:
        delivery = "1920x1080"
    return (
        f"{delivery} 1080p delivery master from the LTX-2.5 DFR production path "
        "using a 1920x1088-class source canvas before the final delivery crop/transcode."
    )


def prepare_delivery(
    source_path: Path,
    *,
    aspect_ratio: str,
    quality: str,
    audio_mode: str,
    prefix: str,
) -> Path:
    quality_path = source_path
    if quality != "preview":
        quality_path = GENERATED_DIR / f"{prefix}-{quality}-{uuid.uuid4().hex}.mp4"
        quality_path = transcode_delivery(
            input_path=source_path,
            output_path=quality_path,
            aspect_ratio=aspect_ratio,
            quality=quality,
        )

    register_current_generated_asset(quality_path.name, metadata={"kind": "delivery_source", "quality": quality})

    if audio_mode == "native":
        return quality_path

    audio_path = GENERATED_DIR / f"{prefix}-{audio_mode}-{uuid.uuid4().hex}.mp4"
    processed = process_audio(quality_path, audio_path, audio_mode)
    register_current_generated_asset(processed.name, metadata={"kind": "delivery", "quality": quality, "audio_mode": audio_mode})
    return processed
