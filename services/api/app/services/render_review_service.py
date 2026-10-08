"""Post-render checks never revoke access to a completed render."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.core.config import settings
from app.services.audio_qc import AudioQCResult, evaluate_scene_audio
from app.services.dialogue_service import build_audio_prompt
from app.services.job_service import register_generated_asset


def qc_report(result=None, *, reason: str = "Quality inspection was not performed.") -> dict:
    if result is None:
        return {"status": "not_checked", "note": reason, "checked_at": None}
    values = result.model_dump()
    values["status"] = ("not_checked" if getattr(result, "not_checked", False) else
                        "unavailable" if result.skipped else "passed" if result.passed else "failed")
    values["checked_at"] = datetime.now(timezone.utc).isoformat()
    return values


def aggregate_qc(reports: list[dict]) -> str:
    statuses = {report.get("status", "not_checked") for report in reports}
    if "failed" in statuses:
        return "failed"
    if "unavailable" in statuses:
        return "unavailable"
    if not statuses or "not_checked" in statuses:
        return "not_checked"
    return "passed"


def pass_flag(status: str) -> bool | None:
    return {"passed": True, "failed": False}.get(status)


def register_render(workspace_id: str | None, path: Path, **metadata) -> None:
    if workspace_id:
        register_generated_asset(workspace_id, path.name, metadata=metadata)


@dataclass
class AudioReview:
    path: Path
    report: dict
    warnings: list[str] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    retake: object | None = None


def review_audio(
    path: Path, *, provider, workspace_id: str | None, scene_prompt: str,
    spoken_script: str, audio_direction: str | None, audio_mode: str,
    duration_seconds: float, seed: int, scene_index: int, allow_retake: bool = True,
    progress: Callable[[str], None] | None = None,
) -> AudioReview:
    if audio_mode == "mute":
        return AudioReview(path, qc_report(reason="Audio was intentionally muted; speech QC is not applicable."))
    if not settings.factory_audio_qc_enabled:
        return AudioReview(path, qc_report(reason="Audio QC is disabled by deployment configuration."))

    def inspect(candidate: Path) -> dict:
        try:
            result = evaluate_scene_audio(
                candidate, scene_prompt=scene_prompt, spoken_script=spoken_script,
                audio_direction=audio_direction,
            )
        except Exception as exc:
            result = AudioQCResult(passed=False, skipped=True, note=f"Audio QC unavailable ({type(exc).__name__}).")
        return qc_report(result)

    if progress:
        progress("Checking dialogue and audio quality...")
    initial = inspect(path)
    review = AudioReview(path, initial, attempts=[{"filename": path.name, **initial}])
    if allow_retake and initial["status"] == "failed" and getattr(provider, "supports_audio_retake", False) and settings.factory_audio_retake_enabled:
        try:
            if progress:
                progress("Correcting audio on the GPU; the existing picture is preserved...")
            # Only the immutable script and sound direction enter the audio repair.
            retake = provider.retake_audio(
                video_path=str(path),
                prompt=build_audio_prompt("", spoken_script, audio_direction),
                duration_seconds=duration_seconds, seed=seed,
            )
            candidate = Path(retake.path)
            if not candidate.is_file() or not candidate.stat().st_size:
                raise RuntimeError("Audio repair produced no video file.")
            register_render(workspace_id, candidate, kind="audio_retake", scene_index=scene_index,
                            audio_qc_status="not_checked")
            review.retake = retake
            review.path = candidate
            if progress:
                progress("Checking the corrected audio...")
            review.report = inspect(candidate)
            review.attempts.append({"filename": candidate.name, **review.report})
        except Exception as exc:
            review.warnings.append(
                f"Scene {scene_index + 1}: audio repair unavailable ({type(exc).__name__}); original video retained for review."
            )
    if review.report["status"] in {"failed", "unavailable"}:
        reason = review.report.get("note") or "; ".join(review.report.get("violations") or []) or "Audio inspection did not pass."
        review.warnings.append(f"Scene {scene_index + 1}: {reason} Video retained for review.")
    register_render(workspace_id, review.path, kind="scene", scene_index=scene_index,
                    audio_qc_status=review.report["status"], audio_qc=review.report,
                    audio_qc_attempts=review.attempts)
    return review
