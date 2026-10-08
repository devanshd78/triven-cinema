import json
import math
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.schemas.generation import EntityLock, Scene
from app.services.continuity_service import (
    compose_continuity_prompt,
    infer_local_entity_locks,
    local_character_bible,
    local_style_bible,
)
from app.services.gemini_service import generate_content
from app.services.dialogue_service import separate_visual_and_script, split_spoken_script


class EntityLockDraft(BaseModel):
    label: str = Field(min_length=1, max_length=64)
    expected_count: int = Field(default=1, ge=1, le=8)
    description: str = Field(default="", max_length=1200)


class SceneDraft(BaseModel):
    title: str = Field(description="Short cinematic title for the scene.")
    prompt: str = Field(
        description=(
            "Detailed generation-ready AI video prompt including subject, action, "
            "environment, camera, lighting and style."
        )
    )
    duration_seconds: int = Field(
        ge=3,
        le=30,
        description="Recommended duration of the scene.",
    )
    visible_entity_counts: dict[str, int] = Field(default_factory=dict)


class ScenePlannerOutput(BaseModel):
    entity_locks: list[EntityLockDraft] = Field(default_factory=list, max_length=12)
    character_bible: str = Field(
        min_length=20,
        max_length=2400,
        description="Immutable recurring-character identity specification.",
    )
    style_bible: str = Field(
        min_length=20,
        max_length=1800,
        description="Immutable visual-style specification shared by every scene.",
    )
    scenes: list[SceneDraft]


@dataclass
class ScenePlanResult:
    scenes: list[Scene]
    source: str
    character_bible: str
    style_bible: str
    entity_locks: list[EntityLock]
    note: str | None = None


def _assign_spoken_script(scenes: list[Scene], script: str) -> list[Scene]:
    scripts = split_spoken_script(script, len(scenes))
    return [scene.model_copy(update={"spoken_script": line,
                                    "prompt": separate_visual_and_script(scene.prompt)[0]})
            for scene, line in zip(scenes, scripts)]


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _limit_words(value: str, limit: int) -> str:
    words = _compact(value).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:") + "."


def _locked_scenes(
    scenes: list[Scene],
    *,
    character_bible: str,
    style_bible: str,
    entity_locks: list[EntityLock],
) -> list[Scene]:
    count = len(scenes)
    return [
        Scene(
            id=scene.id,
            title=scene.title,
            prompt=compose_continuity_prompt(
                scene_prompt=scene.prompt,
                character_bible=character_bible,
                style_bible=style_bible,
                scene_index=index,
                scene_count=count,
                entity_locks=entity_locks,
                visible_entity_counts=scene.visible_entity_counts,
            ),
            duration_seconds=scene.duration_seconds,
            spoken_script=scene.spoken_script,
            visible_entity_counts=scene.visible_entity_counts,
        )
        for index, scene in enumerate(scenes)
    ]


def _story_only(prompt: str) -> str:
    """Prefer the screenplay/story body over style and negative-rule preambles."""
    match = re.search(r"(?im)^#{1,3}\s+STORY\s*$", prompt)
    if match:
        return prompt[match.end() :].strip()
    return prompt.strip()


def _story_units(prompt: str) -> list[tuple[str, str]]:
    story = _story_only(prompt)
    heading = re.compile(r"(?m)^#{1,2}\s+(.+?)\s*$")
    matches = list(heading.finditer(story))
    units: list[tuple[str, str]] = []
    skip_terms = {
        "FINAL TITLE CARD",
        "NEGATIVE GENERATION RULES",
        "GENERATION PRIORITY",
    }

    if matches:
        for index, match in enumerate(matches):
            title = _compact(match.group(1).strip("# "))
            upper = title.upper()
            if any(term in upper for term in skip_terms):
                continue
            end = matches[index + 1].start() if index + 1 < len(matches) else len(story)
            body = story[match.end() : end].strip()
            body = re.sub(r"(?m)^#{1,6}\s+", "", body)
            body = _compact(body)
            if body:
                units.append((title[:96], body))

    if units:
        return units

    paragraphs = [_compact(item) for item in re.split(r"\n\s*\n+", story) if _compact(item)]
    if len(paragraphs) > 1:
        return [(f"Story beat {index + 1}", item) for index, item in enumerate(paragraphs)]

    sentences = [
        _compact(item)
        for item in re.split(r"(?<=[.!?])\s+", _compact(story))
        if _compact(item)
    ]
    return [(f"Story beat {index + 1}", item) for index, item in enumerate(sentences)]


def _distribute_story_units(units: list[tuple[str, str]], scene_count: int) -> list[tuple[str, str]]:
    if not units:
        return [(f"Scene {index + 1}", "Continue the requested story naturally.") for index in range(scene_count)]

    groups: list[tuple[str, str]] = []
    for index in range(scene_count):
        start = math.floor(index * len(units) / scene_count)
        end = math.floor((index + 1) * len(units) / scene_count)
        if end <= start:
            source_index = min(start, len(units) - 1)
            selected = [units[source_index]]
        else:
            selected = units[start:end]
        title = selected[0][0] if selected else f"Scene {index + 1}"
        body = " ".join(item[1] for item in selected)
        groups.append((title, _limit_words(body, 55)))
    return groups


def _direct_style_context(prompt: str, max_words: int = 42) -> str:
    """Extract only user-authored visual-style text for prompt-only Factory mode.

    Direct mode must not ask Gemini to rewrite the request, but a long screenplay
    still needs one small piece of repeated style context so independently rendered
    LTX shots do not drift into different visual media. We only reuse words that
    already exist in the user's prompt.
    """
    match = re.search(r"(?im)^#{1,3}\s+CORE VISUAL STYLE\s*$", prompt)
    if not match:
        return ""
    next_heading = re.search(r"(?m)^#{1,3}\s+.+$", prompt[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(prompt)
    body = re.sub(r"(?m)^[-*]\s+", "", prompt[match.end() : end]).strip()
    return _limit_words(body, max_words)


def _sentence_bounded_excerpt(value: str, word_budget: int) -> str:
    """Keep a source excerpt near a word budget without inventing new wording."""
    compact = _compact(value)
    if len(compact.split()) <= word_budget:
        return compact

    pieces = [
        _compact(piece)
        for piece in re.split(r"(?<=[.!?…])\s+", compact)
        if _compact(piece)
    ]
    selected: list[str] = []
    used = 0
    for piece in pieces:
        words = piece.split()
        if selected and used + len(words) > word_budget:
            break
        if not selected and len(words) > word_budget:
            return _limit_words(piece, word_budget)
        selected.append(piece)
        used += len(words)
        if used >= word_budget:
            break
    return " ".join(selected) if selected else _limit_words(compact, word_budget)


def _single_shot_user_prompt(prompt: str, max_words: int = 320) -> str:
    """Compact a long talking-head prompt without dropping its hard visual constraints.

    For one-shot Creator renders, the old generic 190-word cap could silently lose
    wardrobe, dialogue or lighting instructions. This deterministic selector keeps
    user-authored high-value sentences only; it does not invent or rewrite content.
    """
    compact = _compact(prompt)
    if len(compact.split()) <= max_words:
        return compact
    sentences = [
        _compact(piece)
        for piece in re.split(r"(?<=[.!?])\s+", compact)
        if _compact(piece)
    ]
    critical_terms = (
        "wear", "wardrobe", "sweater", "shirt", "dress", "jacket", "coat", "top", "trouser", "jeans",
        "skin", "face", "hair", "eyes", "camera", "shot", "lens", "desk", "table", "microphone", "monitor",
        "background", "backdrop", "lighting", "light", "audio", "voice", "sound", "says", "say", "dialogue",
    )
    selected: set[int] = set()
    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        if index < 3 or '"' in sentence or any(term in lower for term in critical_terms):
            selected.add(index)
    if sentences:
        selected.add(len(sentences) - 1)

    ordered = [sentences[index] for index in range(len(sentences)) if index in selected]
    result = " ".join(ordered)
    if len(result.split()) <= max_words:
        return result
    return _sentence_bounded_excerpt(result, max_words)


def _direct_story_segments(
    prompt: str,
    scene_count: int,
    *,
    words_per_scene: int = 118,
) -> list[tuple[str, str]]:
    """Build chronological prompt-only scenes from user-authored story sections.

    This is deliberately *not* AI enhancement. It is deterministic orchestration:
    take the user's own story in order, divide it across the requested number of
    LTX shots, and keep source wording rather than feeding the same full manuscript
    to every shot.
    """
    units = _story_units(prompt)
    if not units:
        clean = _compact(prompt)
        return [(f"Scene {index + 1}", clean) for index in range(max(1, scene_count))]

    # If a manuscript has only a few very large sections, split those sections on
    # sentence boundaries first so a multi-shot direct render can still progress.
    expanded: list[tuple[str, str]] = []
    target_unit_words = max(28, words_per_scene // 2)
    for title, body in units:
        words = body.split()
        if len(words) <= target_unit_words * 2:
            expanded.append((title, body))
            continue

        sentences = [
            _compact(piece)
            for piece in re.split(r"(?<=[.!?…])\s+", body)
            if _compact(piece)
        ]
        chunk: list[str] = []
        chunk_words = 0
        part = 1
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if chunk and chunk_words + sentence_words > target_unit_words:
                expanded.append((f"{title} · part {part}", " ".join(chunk)))
                part += 1
                chunk = []
                chunk_words = 0
            chunk.append(sentence)
            chunk_words += sentence_words
        if chunk:
            expanded.append((f"{title} · part {part}" if part > 1 else title, " ".join(chunk)))

    units = expanded or units
    result: list[tuple[str, str]] = []
    for index in range(max(1, scene_count)):
        start = math.floor(index * len(units) / scene_count)
        end = math.floor((index + 1) * len(units) / scene_count)
        if end <= start:
            source_index = min(start, len(units) - 1)
            selected = [units[source_index]]
        else:
            selected = units[start:end]

        per_unit_budget = max(24, words_per_scene // max(1, len(selected)))
        excerpts: list[str] = []
        for title, body in selected:
            excerpt = _sentence_bounded_excerpt(body, per_unit_budget)
            if excerpt:
                excerpts.append(f"{title}: {excerpt}")
        scene_title = selected[0][0] if selected else f"Scene {index + 1}"
        result.append((scene_title[:120], " ".join(excerpts)))
    return result


def _visible_counts_for_beat(beat: str, entity_locks: list[EntityLock]) -> dict[str, int]:
    text = beat.lower()
    counts: dict[str, int] = {}
    for lock in entity_locks:
        label = lock.label.upper()
        label_words = label.replace("_", " ").lower()
        # Named labels are reliable enough for exact local fallback counts. Generic
        # category locks are left unspecified rather than inventing visibility.
        if label not in {"PERSON", "FOX", "DOG", "CAT", "HORSE", "ROBOT", "CAR"}:
            counts[label] = 1 if re.search(rf"\b{re.escape(label_words)}\b", text) else 0
    return counts


def _local_storyboard(
    prompt: str,
    scene_count: int,
    target_duration_seconds: int = 5,
) -> tuple[list[Scene], str, str, list[EntityLock]]:
    """Deterministic preview fallback without copying the whole manuscript per shot.

    Final-quality Factory jobs normally reject this fallback. It remains useful for
    preview/debug workflows and should still preserve obvious named identities.
    """
    character_bible = local_character_bible(prompt)
    style_bible = local_style_bible(prompt)
    entity_locks = infer_local_entity_locks(prompt)
    beats = _distribute_story_units(_story_units(prompt), scene_count)
    scenes: list[Scene] = []

    shot_guidance = [
        "Establish the location and current subject state with restrained cinematic movement.",
        "Continue the established action and geography with a smooth tracking or motivated cut.",
        "Move closer for an emotional or story-detail beat without changing identity or wardrobe.",
        "Advance the story with a clear reveal while preserving the established environment.",
        "Use a calm interaction beat with readable expressions and stable screen geography.",
        "Progress naturally through the established location; do not re-introduce recurring characters.",
        "Build toward the ending with a deliberate composition and controlled movement.",
        "Resolve the sequence with a quiet, visually complete final beat.",
    ]

    for index in range(scene_count):
        title, beat = beats[index]
        guidance = shot_guidance[min(index, len(shot_guidance) - 1)]
        scene_prompt = (
            f"STORY BEAT: {beat} "
            f"SHOT DIRECTION: {guidance} "
            "Keep this as one continuous shot. Describe only what happens during this shot, not the whole story. "
            "AUDIO DIRECTION: use only synchronized ambience, Foley, music, narration or exact dialogue that is "
            "explicitly supported by this story beat; never invent speech, chanting, singing or background conversation. "
            "No subtitles, logos, text overlays or watermarks."
        )
        scenes.append(
            Scene(
                id=index + 1,
                title=(title or f"Scene {index + 1}")[:120],
                prompt=_limit_words(scene_prompt, 80),
                duration_seconds=max(3, min(30, int(target_duration_seconds))),
                visible_entity_counts=_visible_counts_for_beat(beat, entity_locks),
            )
        )

    return (
        _locked_scenes(
            scenes,
            character_bible=character_bible,
            style_bible=style_bible,
            entity_locks=entity_locks,
        ),
        character_bible,
        style_bible,
        entity_locks,
    )


def _extract_json_text(payload: dict) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    text = "".join(str(part.get("text") or "") for part in parts if not part.get("thought")).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    if not text:
        raise RuntimeError("Gemini returned an empty scene plan.")
    return text


def _gemini_storyboard(
    prompt: str,
    scene_count: int,
    aspect_ratio: str,
    target_duration_seconds: int = 5,
) -> tuple[list[Scene], str, str, list[EntityLock]]:
    planner_prompt = f"""
You are the cinematic continuity and scene-planning engine for Triven Cinema.

Convert the user's manuscript into exactly {scene_count} generation-ready LTX-2.5 shots. These shots are
rendered separately and then chained, so identity, geography, audio intent and temporal progression matter
more than decorative detail.

ORIGINAL USER REQUEST:
{prompt}

TARGET ASPECT RATIO:
{aspect_ratio}

Return ONLY valid JSON with this exact top-level shape:
{{
  "entity_locks": [{{"label": "RADHA", "expected_count": 1, "description": "canonical recurring subject"}}],
  "character_bible": "one compact immutable description of every recurring human/creature",
  "style_bible": "one compact immutable visual/style description shared by the whole film",
  "scenes": [
    {{
      "title": "short title",
      "prompt": "scene-specific generation-ready video prompt",
      "duration_seconds": {target_duration_seconds},
      "visible_entity_counts": {{"RADHA": 1, "KRISHNA": 0}}
    }}
  ]
}}

ENTITY/CARDINALITY RULES:
- Create one stable UPPERCASE label for every NAMED recurring physical subject (RADHA, KRISHNA, etc.).
- Never collapse multiple named humans into a generic PERSON lock. RADHA and KRISHNA are separate identities.
- expected_count is the canonical maximum number of physical instances of that identity; normally 1.
- Every scene must provide exact visible_entity_counts for all recurring named entities: 0 when absent, 1 when present.
- Do not count reflections or shadows as additional physical instances.
- If a group such as GOPIS is requested, do not turn members of that group into copies of a named lead.

CHARACTER BIBLE RULES:
- Preserve explicit user-provided identity traits; do not replace them with generic descriptions.
- For recurring humans lock apparent age/stage, face structure, skin tone, eyes, hair, build and recurring identity traits.
- Wardrobe is a hard shot constraint: preserve the user's exact garment type, colors, pattern, material, sleeves/neckline and accessories.
  Never merge or hybridize two outfits. If a shot explicitly changes clothing, the new shot wardrobe overrides earlier/reference clothing.
- Keep each named identity distinct.
- For creatures lock species, size/proportions, colors/markings and distinctive features.
- Keep this bible compact enough to condition every relevant shot; do not paste the whole manuscript into it.

STYLE/ENVIRONMENT RULES:
- Lock the requested visual medium, rendering treatment, palette, materials, lens language, lighting and atmosphere.
- Preserve recurring landmarks and screen geography where the manuscript establishes them.
- Preserve the user's requested style; never silently switch medium between shots.

SCENE RULES:
- Return exactly {scene_count} chronological scenes and cover the full requested runtime/story arc.
- Each scene is ONE continuous shot/beat. NEVER restate or paste the full original story into an individual scene.
- Keep each scene prompt below 80 words so the final continuity-wrapped LTX prompt stays near the recommended 200-word ceiling.
- Describe only the action, environment state, framing, camera movement, expression and lighting needed for that shot.
- For scene 2 onward, recurring subjects are existing characters, not new introductions or alternate versions.
- When the next shot is image-conditioned by the previous approved frame, wording must describe what CHANGES/NEXT ACTION,
  not recreate already-established faces, clothes or bodies.
- Avoid morph transitions through bodies. Prefer natural cuts, camera movement, match cuts, scenery dissolves or environmental transitions.
- Reject AI-looking construction in the plan itself: no unexplained extra straps/buttons/zippers, duplicated props, warped hands, invented text/signage, or unstable set geometry.
- Do not place two copies of the same named identity in one frame unless the manuscript explicitly requests a duplicate.
- Dialogue is managed separately as an immutable user script. Do not invent, quote, summarize or rewrite speech in visual scene prompts.
- No fake Hindi/Sanskrit, random singing, mumbling, speech-like ambience, subtitles, logos or watermarks.
- Do not mention output resolution.
- Do not include any explanation outside the JSON object.
""".strip()

    gemini = generate_content(
        parts=[{"text": planner_prompt}],
        response_mime_type="application/json",
        max_output_tokens=8192,
        timeout_seconds=settings.gemini_timeout_seconds,
        thinking_level=settings.gemini_thinking_level,
    )

    try:
        raw_text = _extract_json_text(gemini.payload)
        parsed = ScenePlannerOutput.model_validate_json(raw_text)
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError("Gemini returned an invalid storyboard JSON response.") from exc

    if len(parsed.scenes) != scene_count:
        raise RuntimeError(
            f"Expected {scene_count} scenes but Gemini returned {len(parsed.scenes)}."
        )

    character_bible = _compact(parsed.character_bible)[:1800]
    style_bible = _compact(parsed.style_bible)[:1000]
    entity_locks = [
        EntityLock(
            label=_compact(lock.label).upper()[:64],
            expected_count=lock.expected_count,
            description=_compact(lock.description)[:1000],
        )
        for lock in parsed.entity_locks
    ]
    scenes = [
        Scene(
            id=index + 1,
            title=_compact(scene.title)[:120] or f"Scene {index + 1}",
            prompt=_limit_words(scene.prompt, 80),
            duration_seconds=max(3, min(30, int(scene.duration_seconds))),
            visible_entity_counts={
                str(k).upper(): max(0, min(8, int(v)))
                for k, v in scene.visible_entity_counts.items()
            },
        )
        for index, scene in enumerate(parsed.scenes)
    ]
    return (
        _locked_scenes(
            scenes,
            character_bible=character_bible,
            style_bible=style_bible,
            entity_locks=entity_locks,
        ),
        character_bible,
        style_bible,
        entity_locks,
    )


def create_prompt_only_plan(
    prompt: str,
    scene_count: int,
    *,
    target_scene_duration_seconds: float | None = None,
    spoken_script: str = "",
) -> ScenePlanResult:
    """Create a chronological Factory plan without Gemini or creative rewriting.

    AI Enhancement OFF means the user's prompt remains the only creative source.
    For multi-shot Factory jobs we deterministically sequence user-authored story
    sections instead of sending the same full manuscript to every LTX shot. This is
    orchestration, not enhancement: no new plot, dialogue, character traits or
    visual ideas are invented here.
    """
    clean_prompt, script = separate_visual_and_script(prompt, spoken_script)
    target_duration = max(3, min(30, int(round(target_scene_duration_seconds or 5))))
    character_bible = local_character_bible(clean_prompt)
    style_bible = local_style_bible(clean_prompt)
    entity_locks = infer_local_entity_locks(clean_prompt)
    if max(1, scene_count) == 1:
        direct_segments = [("Single continuous shot", _single_shot_user_prompt(clean_prompt))]
    else:
        direct_segments = _direct_story_segments(clean_prompt, max(1, scene_count), words_per_scene=150)
    style_context = _direct_style_context(clean_prompt)
    scenes: list[Scene] = []
    for index, (title, source_segment) in enumerate(direct_segments):
        transition = (
            "Continue from the supplied previous frame into this next chronological story segment. "
            "Do not replay an earlier beat, reset to the opening composition, or re-introduce characters already established."
            if index > 0
            else "Render only this opening chronological story segment as one continuous shot."
        )
        pieces = []
        if style_context:
            pieces.append(f"USER STYLE: {style_context}")
        pieces.append(f"USER STORY SEGMENT {index + 1}/{len(direct_segments)}: {source_segment}")
        pieces.append(f"SEQUENCING CONTROL: {transition}")
        scene_prompt = _limit_words(" ".join(pieces), 340 if max(1, scene_count) == 1 else 220)
        scenes.append(
            Scene(
                id=index + 1,
                title=title or f"Direct scene {index + 1}",
                prompt=scene_prompt,
                duration_seconds=target_duration,
                # Direct mode does not pretend to semantically understand whether a
                # name in source text is physically visible, remembered, narrated,
                # or off-screen. Canonical max locks still protect against clones.
                visible_entity_counts={},
            )
        )
    return ScenePlanResult(
        scenes=_assign_spoken_script(scenes, script),
        source="direct",
        character_bible=character_bible,
        style_bible=style_bible,
        entity_locks=entity_locks,
        note=(
            "AI Enhancement is OFF. Gemini planning was skipped. Triven deterministically "
            "sequences only the user's own story sections across LTX shots so each scene "
            "advances instead of replaying the full prompt from the beginning."
        ),
    )


def create_scene_plan(
    prompt: str,
    scene_count: int,
    aspect_ratio: str = "16:9",
    *,
    force_ai: bool = False,
    target_scene_duration_seconds: float | None = None,
    spoken_script: str = "",
) -> ScenePlanResult:
    # Preserve headings, dialogue boundaries and paragraph structure for Gemini.
    # The previous implementation collapsed the full manuscript to one line before
    # planning, which made screenplay structure much harder to recover.
    clean_prompt, script = separate_visual_and_script(prompt, spoken_script)
    target_duration = max(3, min(30, int(round(target_scene_duration_seconds or 5))))

    if scene_count == 1 and not force_ai:
        scenes, character_bible, style_bible, entity_locks = _local_storyboard(
            clean_prompt, 1, target_duration_seconds=target_duration
        )
        return ScenePlanResult(
            scenes=_assign_spoken_script(scenes, script),
            source="direct",
            character_bible=character_bible,
            style_bible=style_bible,
            entity_locks=entity_locks,
            note="Single-scene storyboard created locally without spending a Gemini request.",
        )

    try:
        scenes, character_bible, style_bible, entity_locks = _gemini_storyboard(
            clean_prompt,
            scene_count,
            aspect_ratio,
            target_duration_seconds=target_duration,
        )
        return ScenePlanResult(
            scenes=_assign_spoken_script(scenes, script),
            source="gemini",
            character_bible=character_bible,
            style_bible=style_bible,
            entity_locks=entity_locks,
            note="Storyboard generated by Gemini with shared identity, entity-count, audio and style locks.",
        )
    except Exception as exc:  # preview/debug reliability boundary
        scenes, character_bible, style_bible, entity_locks = _local_storyboard(
            clean_prompt, scene_count, target_duration_seconds=target_duration
        )
        return ScenePlanResult(
            scenes=_assign_spoken_script(scenes, script),
            source="fallback",
            character_bible=character_bible,
            style_bible=style_bible,
            entity_locks=entity_locks,
            note=(
                f"{str(exc)} Triven used the bounded local PREVIEW fallback. "
                "Final-quality Factory jobs reject this fallback unless explicitly enabled."
            ),
        )
