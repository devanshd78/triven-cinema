import argparse
import subprocess
import uuid
from pathlib import Path


LTX_REPO = Path("/opt/LTX-2")
LTX_PYTHON = LTX_REPO / ".venv" / "bin" / "python"


def build_retake_subprocess_command(
    *,
    input_path: Path,
    output_path: Path,
    prompt: str,
    start_time: float,
    end_time: float,
    seed: int,
    python_executable: Path = LTX_PYTHON,
) -> list[str]:
    """Build the command that runs Retake inside the LTX uv virtualenv.

    The Modal function itself runs under the image's outer Python interpreter,
    while `uv sync --extra natten` installs torch and the LTX packages into
    `/opt/LTX-2/.venv`. Running this module with that interpreter keeps Retake
    on the exact same tested LTX/PyTorch environment as video generation.
    """
    return [
        str(python_executable),
        str(Path(__file__).resolve()),
        "--input-path",
        str(input_path),
        "--output-path",
        str(output_path),
        "--prompt",
        prompt,
        "--start-time",
        str(max(0.0, float(start_time))),
        "--end-time",
        str(max(0.0, float(end_time))),
        "--seed",
        str(int(seed)),
    ]


def retake_audio_only(
    *,
    input_path: Path,
    output_path: Path,
    prompt: str,
    start_time: float,
    end_time: float,
    seed: int,
) -> None:
    """Regenerate audio while freezing video in the LTX project environment.

    Do not import torch/LTX packages in Modal's outer interpreter. They live in
    the uv environment created by `uv sync` at image build time. The previous
    implementation imported torch here and failed with `No module named 'torch'`.
    """
    if not LTX_PYTHON.exists():
        raise RuntimeError(
            "LTX Python environment is missing at /opt/LTX-2/.venv/bin/python. "
            "Redeploy the current Modal image with `modal deploy modal/app.py`."
        )

    repaired_path = output_path.with_name(f".retake-{uuid.uuid4().hex}.mp4")
    command = build_retake_subprocess_command(
        input_path=input_path,
        output_path=repaired_path,
        prompt=prompt,
        start_time=start_time,
        end_time=end_time,
        seed=seed,
    )
    try:
        process = subprocess.run(command, cwd=LTX_REPO, capture_output=True, text=True)
        if process.returncode != 0:
            details = (process.stderr or process.stdout or "unknown Retake failure")[-8000:]
            raise RuntimeError("LTX audio Retake failed:\n" + details)
        # Retake freezes video latents, but its VAE round trip changes pixels.
        # Keep the original compressed picture; use only the repaired audio.
        mux = subprocess.run(build_audio_retake_mux_command(input_path, repaired_path, output_path), capture_output=True, text=True)
        if mux.returncode or not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Unable to mux repaired audio onto original video: " + mux.stderr[-2000:])
    finally:
        repaired_path.unlink(missing_ok=True)


def build_audio_retake_mux_command(source_path: Path, repaired_path: Path, output_path: Path) -> list[str]:
    return ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source_path), "-i", str(repaired_path),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
            "-movflags", "+faststart", str(output_path)]


def _retake_audio_in_ltx_env(
    *,
    input_path: Path,
    output_path: Path,
    prompt: str,
    start_time: float,
    end_time: float,
    seed: int,
) -> None:
    """Actual Retake implementation; executed by /opt/LTX-2/.venv/bin/python."""
    import torch
    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_pipelines.retake import RetakePipeline
    from ltx_pipelines.utils.media_io import (
        encode_video,
        get_videostream_metadata,
        resolve_hdr_color_space,
        vae_dtype_for_hdr,
    )
    from ltx_pipelines.utils.model_paths import ModelPaths

    from models import AUDIO_VAE, TEXT_ENCODER, TRANSFORMER, VIDEO_VAE_DIFFUSION

    model_paths = ModelPaths.from_split(
        transformer_path=str(TRANSFORMER),
        text_encoder_path=str(TEXT_ENCODER),
        video_vae_path=str(VIDEO_VAE_DIFFUSION),
        audio_vae_path=str(AUDIO_VAE),
    )
    source = get_videostream_metadata(str(input_path))
    hdr = resolve_hdr_color_space(video_paths=[str(input_path)], hdr=None)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    pipeline = RetakePipeline(
        model_paths=model_paths,
        loras=[],
        distilled=True,
    )
    with torch.inference_mode():
        result = pipeline(
            video_path=str(input_path),
            prompt=prompt,
            start_time=max(0.0, float(start_time)),
            end_time=min(float(end_time), max(0.01, (source.frames - 1) / source.fps)),
            seed=int(seed),
            regenerate_video=False,
            regenerate_audio=True,
            vae_dtype=vae_dtype,
            color_space=hdr,
            tiling_config=AUTO_TILING,
            max_batch_size=1,
        )
        encode_video(
            video=result.video,
            fps=int(source.fps),
            audio=result.audio,
            output_path=str(output_path),
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
            color_space=hdr,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LTX audio-only Retake inside the LTX uv environment.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _retake_audio_in_ltx_env(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        prompt=args.prompt,
        start_time=args.start_time,
        end_time=args.end_time,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
