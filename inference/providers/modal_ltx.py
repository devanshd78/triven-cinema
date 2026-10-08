import logging
import os
import time
import uuid
from pathlib import Path

import modal
from dotenv import load_dotenv

from inference.providers.base import VideoGenerationResult, VideoProvider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
GENERATED_DIR = PROJECT_ROOT / "storage" / "generated"
LOGGER = logging.getLogger("triven.modal_ltx")


class ModalLTXProvider(VideoProvider):
    """Client for the self-hosted LTX-2.5 Modal app in modal/app.py."""

    name = "modal-ltx-2.5"
    supports_native_long_video = True
    supports_audio_retake = True
    _preflight_cache: dict[tuple, tuple[float, dict]] = {}

    def __init__(self):
        self.app_name = os.getenv("MODAL_APP_NAME", "triven-cinema-ltx")
        self.function_name = os.getenv("MODAL_FUNCTION_NAME", "generate_video")
        self.audio_retake_function_name = os.getenv("MODAL_AUDIO_RETAKE_FUNCTION_NAME", "retake_audio")
        self.preflight_function_name = os.getenv("MODAL_PREFLIGHT_FUNCTION_NAME", "preflight")
        self.ltx_repo_ref = os.getenv("TRIVEN_LTX_REPO_REF", "v1.4.2").strip() or "v1.4.2"
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    def preflight(self, *, render_mode="distilled", realism_profile="standard", element_reference_required=False,
                  operation="generate", decoder="conv", force_refresh=False) -> dict:
        arguments = dict(render_mode=render_mode, realism_profile=realism_profile,
                         element_reference_required=bool(element_reference_required), operation=operation, decoder=decoder)
        key = (self.app_name, self.preflight_function_name, self.ltx_repo_ref, *arguments.values())
        cached = self._preflight_cache.get(key)
        if not force_refresh and cached and time.monotonic() - cached[0] < 60:
            return cached[1]
        try:
            remote = modal.Function.from_name(self.app_name, self.preflight_function_name)
            manifest = remote.remote(**arguments)
        except Exception as exc:
            detail = (str(exc) or type(exc).__name__).strip().replace("\n", " ")[:500]
            if "workspace" in detail.lower() and "disabled" in detail.lower():
                raise RuntimeError("Modal workspace is disabled. Restore workspace access in the Modal dashboard "
                                   "or contact Modal support before retrying. The readiness check could not run.") from exc
            raise RuntimeError("Modal inference preflight failed before GPU allocation. Verify Modal authentication and "
                               f"deploy the current compatible worker with `modal deploy modal/app.py`. App: {self.app_name}. {detail}") from exc
        if not isinstance(manifest, dict) or manifest.get("protocol_version") != 2:
            raise RuntimeError("Modal worker protocol is incompatible. Deploy the current worker with `modal deploy modal/app.py` before generating.")
        if manifest.get("ltx_repo_ref") != self.ltx_repo_ref:
            raise RuntimeError(f"Modal worker LTX revision mismatch: expected {self.ltx_repo_ref}, "
                               f"received {manifest.get('ltx_repo_ref') or 'unknown'}. Align TRIVEN_LTX_REPO_REF "
                               "in the API environment and worker deployment; recreate the API container after changing its .env.")
        if not manifest.get("ready"):
            raise RuntimeError("Modal inference is not ready: " + "; ".join(str(item) for item in manifest.get("errors", ["Unknown preflight error"])))
        self._preflight_cache[key] = (time.monotonic(), manifest)
        return manifest

    @staticmethod
    def _write_video(video_bytes: bytes, prefix: str) -> Path:
        filename = f"{prefix}-{uuid.uuid4().hex}.mp4"
        destination = GENERATED_DIR / filename
        temporary = destination.with_suffix(".mp4.part")
        temporary.write_bytes(video_bytes)
        temporary.replace(destination)
        return destination

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        duration_seconds: float,
        seed: int,
        decoder: str,
        enhance_prompt: bool = False,
        render_mode: str = "distilled",
        reference_image_path: str | None = None,
        reference_strength: float = 0.95,
        element_reference_sheet_path: str | None = None,
        element_reference_strength: float = 1.0,
        realism_profile: str = "standard",
    ) -> VideoGenerationResult:
        started = time.perf_counter()

        reference_bytes: bytes | None = None
        reference_suffix = ".png"
        element_reference_sheet_bytes: bytes | None = None
        if reference_image_path:
            path = Path(reference_image_path)
            if not path.exists():
                raise FileNotFoundError(f"Continuity reference frame not found: {path.name}")
            reference_bytes = path.read_bytes()
            reference_suffix = path.suffix.lower() or ".png"

        if element_reference_sheet_path:
            sheet = Path(element_reference_sheet_path)
            if not sheet.exists():
                raise FileNotFoundError(f"Element reference sheet not found: {sheet.name}")
            element_reference_sheet_bytes = sheet.read_bytes()

        readiness_started = time.perf_counter()
        self.preflight(render_mode=render_mode, realism_profile=realism_profile,
                       element_reference_required=bool(element_reference_sheet_bytes), decoder=decoder)
        readiness_seconds = time.perf_counter() - readiness_started
        remote_started = time.perf_counter()

        try:
            remote_function = modal.Function.from_name(self.app_name, self.function_name)
            result = remote_function.remote(
                prompt=prompt,
                width=width,
                height=height,
                duration_seconds=duration_seconds,
                seed=seed,
                decoder=decoder,
                enhance_prompt=enhance_prompt,
                render_mode=render_mode,
                reference_image_bytes=reference_bytes,
                reference_image_suffix=reference_suffix,
                reference_strength=float(reference_strength),
                element_reference_sheet_bytes=element_reference_sheet_bytes,
                element_reference_strength=float(element_reference_strength),
                realism_profile=realism_profile,
            )
        except Exception as exc:
            LOGGER.exception(
                "Modal LTX remote invocation failed app=%s function=%s",
                self.app_name,
                self.function_name,
            )
            detail = (str(exc) or repr(exc)).strip().replace("\n", " ")[:600]
            raise RuntimeError(
                "Modal LTX generation failed. Redeploy the current worker with "
                "`modal deploy modal/app.py`, then inspect `modal app logs "
                f"{self.app_name}` if it still fails. Remote error: {detail}"
            ) from exc

        video_bytes = result.get("video_bytes")
        if not video_bytes:
            raise RuntimeError("Modal returned no video bytes.")
        destination = self._write_video(video_bytes, "ltx-modal")
        base_destination = None
        if result.get("base_video_bytes"):
            try:
                base_destination = self._write_video(result["base_video_bytes"], "ltx-base")
            except OSError:
                LOGGER.warning("Could not save pre-refinement alternative; the completed render is preserved.")

        wall_elapsed = time.perf_counter() - started
        elapsed = float(result.get("render_seconds") or 0.0) or wall_elapsed
        remote_seconds = time.perf_counter() - remote_started
        timings_seconds = {**(result.get("timings_seconds") or {}),
                           "readiness_check": round(readiness_seconds, 3),
                           "queue_startup_transfer": round(max(0.0, remote_seconds - elapsed), 3)}
        return VideoGenerationResult(
            filename=destination.name,
            path=str(destination),
            seed=int(result.get("seed", seed)),
            render_details=str(result.get("render_details", f"{width}x{height} · Modal LTX-2.5")),
            render_seconds=elapsed,
            prompt=str(result.get("prompt", prompt)),
            provider=self.name,
            gpu=str(result.get("gpu") or "Modal GPU"),
            wall_seconds=wall_elapsed,
            reference_conditioned=bool(result.get("reference_conditioned", False)),
            chunk_count=max(1, int(result.get("chunk_count") or 1)),
            render_mode=str(result.get("render_mode") or render_mode),
            realism_profile=str(result.get("realism_profile") or realism_profile),
            detail_refined=bool(result.get("detail_refined", False)),
            timings_seconds=timings_seconds,
            base_path=str(base_destination) if base_destination else None,
        )

    def retake_audio(
        self,
        *,
        video_path: str,
        prompt: str,
        duration_seconds: float,
        seed: int,
    ) -> VideoGenerationResult:
        source = Path(video_path)
        if not source.exists():
            raise FileNotFoundError(f"Audio Retake source video not found: {source.name}")

        started = time.perf_counter()
        self.preflight(operation="retake_audio", decoder="diffusion")
        try:
            remote_function = modal.Function.from_name(self.app_name, self.audio_retake_function_name)
            result = remote_function.remote(
                video_bytes=source.read_bytes(),
                prompt=prompt,
                duration_seconds=float(duration_seconds),
                seed=int(seed),
            )
        except Exception as exc:
            LOGGER.exception("Modal LTX audio Retake failed app=%s", self.app_name)
            detail = (str(exc) or repr(exc)).strip().replace("\n", " ")[:600]
            raise RuntimeError(f"Modal LTX audio Retake failed: {detail}") from exc

        video_bytes = result.get("video_bytes")
        if not video_bytes:
            raise RuntimeError("Modal audio Retake returned no video bytes.")
        destination = self._write_video(video_bytes, "ltx-audio-retake")
        wall_elapsed = time.perf_counter() - started
        elapsed = float(result.get("render_seconds") or 0.0) or wall_elapsed
        return VideoGenerationResult(
            filename=destination.name,
            path=str(destination),
            seed=int(result.get("seed", seed)),
            render_details=str(result.get("render_details") or "LTX Retake · audio-only"),
            render_seconds=elapsed,
            prompt=str(result.get("prompt", prompt)),
            provider=self.name,
            gpu=str(result.get("gpu") or "Modal GPU"),
            wall_seconds=wall_elapsed,
            reference_conditioned=True,
            chunk_count=1,
            render_mode="audio-retake",
        )
