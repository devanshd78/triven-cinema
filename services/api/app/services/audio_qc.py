import base64
import json
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.continuity_service import compact_text
from app.services.gemini_service import generate_content


class MissingAudioError(RuntimeError):
    pass


class AudioQCResult(BaseModel):
    passed: bool = True
    gibberish_detected: bool = False
    unintended_speech_detected: bool = False
    dialogue_mismatch_detected: bool = False
    violations: list[str] = Field(default_factory=list)
    note: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    skipped: bool = False


def _skipped(note: str) -> AudioQCResult:
    return AudioQCResult(passed=False, skipped=True, note=note, confidence=0.0)


def _extract_json_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini audio QC returned no candidates.")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    text = "".join(
        str(part.get("text") or "")
        for part in parts
        if not part.get("thought")
    ).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    if not text:
        raise RuntimeError("Gemini audio QC returned an empty response.")
    return text


def _extract_audio(video_path: Path, destination: Path) -> None:
    process = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=max(30, settings.ffmpeg_timeout_seconds),
    )
    if process.returncode != 0 or not destination.exists() or destination.stat().st_size < 1024:
        detail = (process.stderr or "audio extraction failed")[-600:].replace("\n", " ")
        if "does not contain any stream" in detail or "matches no streams" in detail:
            raise MissingAudioError("The generated clip contains no audio stream.")
        raise RuntimeError(detail)


def evaluate_scene_audio(
    video_path: Path,
    *,
    scene_prompt: str,
    audio_direction: str | None = None,
    spoken_script: str = "",
) -> AudioQCResult:
    """Reject gibberish / invented speech before final mastering.

    LTX native audio is valuable for synchronized ambience and short speech, but a
    loudness pass cannot make unintelligible vocals correct. This gate listens to
    the generated track and compares it with the requested scene/audio intent.
    """
    if not settings.factory_audio_qc_enabled:
        return _skipped("Audio semantic QC is disabled by deployment configuration.")
    if not settings.gemini_api_key:
        return _skipped("Gemini is not configured; semantic audio QC could not run.")

    instruction = f"""
You are Triven Cinema's strict generated-audio quality inspector.
Listen to the attached WAV extracted from ONE generated video scene.

VISUAL CONTEXT (never dialogue):
{compact_text(scene_prompt, 1000)}

IMMUTABLE SPOKEN SCRIPT — every word must be present exactly once, in order:
{spoken_script or 'No spoken dialogue was requested. Do not invent speech.'}

EXTRA AUDIO DIRECTION:
{compact_text(audio_direction or 'No extra audio direction.', 800)}

FAIL the audio if any of these are clearly present:
- gibberish, fake-language, garbled, slurred, chopped, unintelligible, or speech-like vocal noise;
- background people talking when the scene did not request speech;
- chanting or singing not explicitly requested;
- spoken words that differ from the IMMUTABLE SPOKEN SCRIPT, or omitted/repeated lines;
- overlapping duplicate voices that make intended dialogue unintelligible.

Do NOT fail ordinary non-speech ambience, Foley, music, flute, wind, water, birds, cloth movement, breaths, or intentional silence.
If the scene requests no speech, speech-like human vocalization is a failure even when words cannot be understood.
Be conservative about tiny compression artifacts, but strict about human-vocal corruption.

Return ONLY JSON:
{{
  "passed": true,
  "gibberish_detected": false,
  "unintended_speech_detected": false,
  "dialogue_mismatch_detected": false,
  "violations": [],
  "note": "short reason",
  "confidence": 0.95,
  "skipped": false
}}
""".strip()

    try:
        with tempfile.TemporaryDirectory(prefix=".triven-audio-qc-", dir=video_path.parent) as tmp:
            wav_path = Path(tmp) / "scene.wav"
            _extract_audio(video_path, wav_path)
            parts = [
                {"text": instruction},
                {
                    "inlineData": {
                        "mimeType": "audio/wav",
                        "data": base64.b64encode(wav_path.read_bytes()).decode("ascii"),
                    }
                },
            ]
            response = generate_content(
                parts=parts,
                max_output_tokens=1000,
                timeout_seconds=max(settings.gemini_timeout_seconds, 30.0),
                total_timeout_seconds=settings.quality_check_retry_budget_seconds,
                max_attempts_per_model=1,
                thinking_level="low",
            )
            payload = json.loads(_extract_json_text(response.payload))
            required = {"passed", "gibberish_detected", "unintended_speech_detected",
                        "dialogue_mismatch_detected", "violations", "note", "confidence"}
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError("QC provider returned an incomplete verdict.")
            parsed = AudioQCResult.model_validate(payload, strict=True)
            if parsed.skipped or not parsed.note.strip() or parsed.confidence <= 0:
                raise ValueError("QC provider returned an unverified verdict.")
    except MissingAudioError as exc:
        return AudioQCResult(passed=False, dialogue_mismatch_detected=bool(spoken_script),
                             violations=[str(exc)], note=str(exc), confidence=1.0)
    except Exception as exc:
        return _skipped(f"Audio QC unavailable ({str(exc)[:180]}).")

    if (
        parsed.gibberish_detected
        or parsed.unintended_speech_detected
        or parsed.dialogue_mismatch_detected
        or parsed.violations
    ):
        parsed.passed = False
    return parsed
