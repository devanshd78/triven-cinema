import base64
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.generation import EntityLock
from app.services.continuity_service import compact_text, normalize_entity_locks
from app.services.gemini_service import generate_content
from app.services.video_combiner import extract_qc_frames


class ContinuityQCResult(BaseModel):
    passed: bool = True
    duplicate_detected: bool = False
    identity_drift_detected: bool = False
    artifact_detected: bool = False
    wardrobe_mismatch_detected: bool = False
    observed_max_counts: dict[str, int] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    identity_violations: list[str] = Field(default_factory=list)
    artifact_violations: list[str] = Field(default_factory=list)
    wardrobe_violations: list[str] = Field(default_factory=list)
    note: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    skipped: bool = False
    not_checked: bool = False


def _extract_json_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini continuity QC returned no candidates.")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    text_parts = [
        str(part.get("text") or "")
        for part in parts
        if not part.get("thought")
    ]
    text = "".join(text_parts).strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    if not text:
        raise RuntimeError("Gemini continuity QC returned an empty response.")
    return text


def _skipped(note: str, *, not_checked: bool = False) -> ContinuityQCResult:
    return ContinuityQCResult(passed=False, skipped=True, not_checked=not_checked, note=note, confidence=0.0)


def _image_part(path: Path) -> dict:
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    return {
        "inlineData": {
            "mimeType": mime,
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


def evaluate_scene_cardinality(
    video_path: Path,
    *,
    entity_locks: list[EntityLock] | list[dict] | None,
    visible_entity_counts: dict[str, int] | None,
    character_bible: str | None,
    scene_prompt: str,
    qc_mode: str = "auto",
    reference_frame_path: Path | None = None,
    canonical_reference_paths: list[tuple[str, Path]] | None = None,
) -> ContinuityQCResult:
    """Inspect frames for duplicate subjects *and* recurring-character identity drift.

    When a previous approved frame is available it becomes the identity reference for
    the current shot. This makes QC answer both questions that matter for narrative
    continuity: "how many?" and "is this still the same character?".
    """
    if qc_mode == "off":
        return _skipped("Continuity vision QC disabled for this request.", not_checked=True)
    if not settings.continuity_vision_qc_enabled:
        return _skipped("Continuity vision QC is disabled by deployment configuration.", not_checked=True)
    if not settings.gemini_api_key:
        return _skipped("Gemini is not configured; prompt/cardinality guard remains active.")

    locks = normalize_entity_locks(entity_locks)
    lock_lines = [
        f"{lock.label}: canonical maximum {lock.expected_count}; {lock.description}"
        for lock in locks
    ]
    if visible_entity_counts:
        exact = ", ".join(
            f"{str(label).upper()}={max(0, int(count))}"
            for label, count in visible_entity_counts.items()
        )
    else:
        exact = "No exact per-shot counts supplied; enforce canonical maximums and reject obvious clones."

    has_reference = bool(reference_frame_path and reference_frame_path.exists())
    canonical_refs = [
        (label, path) for label, path in (canonical_reference_paths or [])
        if path and path.exists()
    ]
    reference_instruction = []
    if canonical_refs:
        reference_instruction.append(
            "The first labeled reference images are CANONICAL ELEMENT REFERENCES. They define who/what the recurring Elements must look like and outrank incidental drift in previous generated frames."
        )
    if has_reference:
        reference_instruction.append(
            "A PREVIOUS APPROVED CONTINUITY FRAME is also attached. Use it for pose, wardrobe state, geography and motion continuity, but do not let gradual drift override the canonical Element references."
        )
    if not reference_instruction:
        reference_instruction.append(
            "No external identity reference is available for this first shot; evaluate cardinality, anatomy, and consistency within the sampled clip."
        )
    prompt = f"""
You are Triven Cinema's strict visual continuity inspector.
The generated clip is represented by sampled frames attached after this instruction.
{' '.join(reference_instruction)}

CANONICAL ENTITY LOCKS:
{chr(10).join(lock_lines) if lock_lines else 'No structured locks available. Use the character bible and reject obvious duplicate copies of the same recurring subject.'}

EXPECTED COUNTS FOR THIS SHOT:
{exact}

CHARACTER BIBLE:
{compact_text(character_bible, 2200)}

SCENE INTENT:
{compact_text(scene_prompt, 1800)}

QC RULES:
- Count physical subjects WITHIN each individual generated frame; never add counts across different frames.
- A recurring subject must not appear as twins, mirrored physical duplicates, extra bodies, extra heads/faces, split bodies, fused people, or ghost clones.
- Do not count a normal shadow or a clearly readable reflection as another physical subject.
- For an expected count of 1, two simultaneously visible physical instances is a failure.
- If canonical Element references exist, recurring named characters/props/locations must match those references first for identity: recognizable face geometry, age band, skin tone, hair/hairline, body proportions, object design, or location landmarks as applicable.
- IMPORTANT wardrobe rule: when SCENE INTENT contains WARDROBE AUTHORITY or says scene wardrobe overrides the identity reference, the canonical Element's clothes are NOT authoritative. Judge the generated clothing against the scene's requested garment, colors, pattern, material and construction instead. Reject hybrid/blended reference clothing.
- Otherwise, if a previous approved reference exists, recurring named characters should preserve the same wardrobe state unless the scene explicitly calls for a justified change.
- A face replacement, unexplained body redesign, one named character turning into another, or strong identity drift is a failure.
- Artifact failure examples: facial melting, eye/teeth deformation, extra/fused fingers or limbs, warped hands, clothing seams/patterns morphing, random straps/buttons/zippers, object duplication, floating geometry, unstable desk/microphone/set geometry, or unrequested glyph/text noise.
- Inspect the EARLIEST generated sample especially carefully. A bad first seconds phase, frozen reference hold, identity redesign, or abrupt wardrobe mutation is a failure even if later samples recover.
- Ignore tiny unrelated background strangers unless they duplicate a locked recurring subject.
- Be conservative about ordinary motion blur, pose changes, expression changes, lighting, and camera perspective. Only flag meaningful identity or artifact failures.
- CRITICAL SOURCE SEPARATION: CANONICAL ELEMENT REFERENCES and PREVIOUS APPROVED CONTINUITY REFERENCE are INPUT photographs, NOT frames of the generated video. They may themselves show multiple views, a collage, or a reference sheet. NEVER report split-screen, duplicate subjects, static frames, or identity drift based on those INPUT images. Only GENERATED CLIP SAMPLE images are evidence of a generated-video artifact. If a sampled generated frame truly contains a split-screen/contact sheet, report it; otherwise do not.

Return ONLY JSON with exactly this shape:
{{
  "passed": true,
  "duplicate_detected": false,
  "identity_drift_detected": false,
  "artifact_detected": false,
  "wardrobe_mismatch_detected": false,
  "observed_max_counts": {{"ENTITY": 1}},
  "violations": [],
  "identity_violations": [],
  "artifact_violations": [],
  "wardrobe_violations": [],
  "note": "short reason",
  "confidence": 0.95,
  "skipped": false
}}
""".strip()

    try:
        with tempfile.TemporaryDirectory(prefix=".triven-qc-", dir=video_path.parent) as tmp:
            frames = extract_qc_frames(
                video_path,
                Path(tmp),
                prefix="continuity-qc",
                positions=(0.02, 0.18, 0.45, 0.72, 0.94)[: max(1, min(5, settings.continuity_qc_max_frames))],
            )
            if not frames:
                return _skipped("No QC frames could be extracted from the generated clip.")

            parts: list[dict] = [{"text": prompt}]
            if canonical_refs:
                parts.append({"text": "CANONICAL ELEMENT REFERENCES:"})
                for label, path in canonical_refs:
                    parts.append({"text": f"CANONICAL {label}:"})
                    parts.append(_image_part(path))
            if has_reference and reference_frame_path is not None:
                parts.append({"text": "PREVIOUS APPROVED CONTINUITY REFERENCE:"})
                parts.append(_image_part(reference_frame_path))
            if canonical_refs or has_reference:
                parts.append({"text": "GENERATED CLIP SAMPLES:"})
            for sample_index, frame in enumerate(frames, start=1):
                parts.append({"text": f"GENERATED CLIP SAMPLE {sample_index}/{len(frames)} — judge actual video defects in this image only:"})
                parts.append(_image_part(frame))

            response = generate_content(
                parts=parts,
                max_output_tokens=1400,
                timeout_seconds=settings.continuity_qc_timeout_seconds,
                thinking_level="low",
            )
            payload = json.loads(_extract_json_text(response.payload))
            required = {"passed", "duplicate_detected", "identity_drift_detected", "artifact_detected",
                        "wardrobe_mismatch_detected", "violations", "identity_violations",
                        "artifact_violations", "wardrobe_violations", "note", "confidence"}
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError("QC provider returned an incomplete verdict.")
            parsed = ContinuityQCResult.model_validate(payload, strict=True)
            if parsed.skipped or parsed.not_checked or not parsed.note.strip() or parsed.confidence <= 0:
                raise ValueError("QC provider returned an unverified verdict.")
    except Exception as exc:
        return _skipped(f"Continuity vision QC unavailable ({type(exc).__name__}: {str(exc)[:140]}).")

    # Never trust a model-produced pass if the same payload reports a hard violation.
    if (
        parsed.duplicate_detected
        or parsed.identity_drift_detected
        or parsed.artifact_detected
        or parsed.wardrobe_mismatch_detected
        or parsed.violations
        or parsed.identity_violations
        or parsed.artifact_violations
        or parsed.wardrobe_violations
    ):
        parsed.passed = False
    return parsed
