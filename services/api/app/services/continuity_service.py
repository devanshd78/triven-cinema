import re
from collections.abc import Iterable

from app.schemas.generation import EntityLock


CONTINUITY_HEADER = "TRIVEN CONTINUITY LOCK"

# These are deliberately phrased as generation constraints, not as a separate
# "negative prompt" because the current LTX distilled CLI accepts one text prompt.
ANTI_DUPLICATION_CONSTRAINTS = (
    "Never create duplicate copies of a recurring subject. No clone, twin, mirrored duplicate, "
    "second copy, extra person, extra animal, extra face, extra head, extra torso, fused duplicate, "
    "ghost duplicate, reflection that looks like a second physical subject, or accidental crowding. "
    "A mirror/window reflection is allowed only when the story explicitly asks for one and it must "
    "read clearly as a reflection, never as another physical character."
)

COMMON_ENTITY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PERSON", ("person", "human", "man", "woman", "boy", "girl", "adventurer", "traveler", "traveller", "character")),
    ("FOX", ("fox",)),
    ("DOG", ("dog", "puppy")),
    ("CAT", ("cat", "kitten")),
    ("HORSE", ("horse",)),
    ("ROBOT", ("robot", "android")),
    ("CAR", ("car", "vehicle", "sports car")),
)

NON_ENTITY_HEADINGS = {
    "AUDIO", "AUDIO DESIGN", "CINEMATIC LANGUAGE", "CRITICAL CONTINUITY RULES",
    "ENVIRONMENT", "FINAL SEQUENCE", "FINAL TITLE CARD", "GENERATION PRIORITY",
    "MUSIC", "NARRATION", "NARRATION VOICE", "NARRATOR", "NEGATIVE GENERATION RULES",
    "STORY", "STYLE", "VISUAL STYLE", "CORE VISUAL STYLE", "PASSAGE OF TIME",
}


def _named_entity_labels(prompt: str) -> list[str]:
    """Extract explicit recurring character names without treating story headings as people."""
    candidates: list[str] = []

    # Strongest signal: an explicit permanent character identity section.
    for match in re.finditer(
        r"(?im)^#{1,5}\s+([A-Z][A-Z0-9 _'-]{1,32}?)\s+[—-]\s+PERMANENT\s+CHARACTER\s+IDENTITY\s*$",
        prompt,
    ):
        candidates.append(match.group(1).strip())

    # Screenplay speaker headings (### RADHA / ### KRISHNA) are also reliable,
    # while narrative headings like THE QUESTION or FIRST DIALOGUE are not.
    for match in re.finditer(r"(?m)^#{2,5}\s+([A-Z][A-Z'-]{2,24})\s*$", prompt):
        candidates.append(match.group(1).strip())

    for match in re.finditer(r"\bnamed\s+([A-Z][a-z]{2,24})\b", prompt):
        candidates.append(match.group(1))

    result: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        label = _safe_label(raw)
        if label in NON_ENTITY_HEADINGS or label == "NARRATOR":
            continue
        if len(label) < 3 or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result[:8]


def _identity_context(prompt: str, label: str, limit: int = 700) -> str:
    pretty = label.replace("_", " ")
    # Prefer the dedicated character identity section and retain deeper nested
    # facial/costume headings. Stop only at a heading of the same or higher level.
    header = re.search(
        rf"(?im)^(#{{1,2}})\s+{re.escape(pretty)}\s+[—-]\s+PERMANENT\s+CHARACTER\s+IDENTITY\s*$",
        prompt,
    )
    if header:
        level = len(header.group(1))
        tail = prompt[header.end() :]
        next_heading = re.search(rf"(?m)^#{{1,{level}}}\s+", tail)
        body = tail[: next_heading.start()] if next_heading else tail
        context = compact_text(body, limit)
        if context:
            return context
    match = re.search(rf"(?is).{{0,220}}\b{re.escape(pretty)}\b.{{0,420}}", prompt)
    return compact_text(match.group(0), limit) if match else ""


def compact_text(value: str | None, limit: int = 2800) -> str:
    if not value:
        return ""
    compacted = re.sub(r"\s+", " ", value).strip()
    return compacted[:limit]


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().upper()).strip("_")
    return label[:64] or "ENTITY"


def normalize_entity_locks(entity_locks: Iterable[EntityLock | dict] | None) -> list[EntityLock]:
    result: list[EntityLock] = []
    seen: set[str] = set()
    for raw in entity_locks or []:
        try:
            lock = raw if isinstance(raw, EntityLock) else EntityLock.model_validate(raw)
        except Exception:
            continue
        label = _safe_label(lock.label)
        if label in seen:
            continue
        seen.add(label)
        result.append(
            EntityLock(
                label=label,
                expected_count=max(1, min(8, int(lock.expected_count))),
                description=compact_text(lock.description, 1200),
            )
        )
    return result[:12]


def infer_local_entity_locks(prompt: str) -> list[EntityLock]:
    """Conservative local entity locks with explicit named-character support."""
    raw = prompt or ""
    text = compact_text(raw, 12000).lower()
    locks: list[EntityLock] = []
    used_terms: set[str] = set()

    named = _named_entity_labels(raw)
    for label in named:
        description = _identity_context(raw, label)
        locks.append(
            EntityLock(
                label=label,
                expected_count=1,
                description=(
                    description
                    or f"Named recurring character {label.replace('_', ' ').title()}; preserve one canonical identity exactly."
                ),
            )
        )

    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for label, terms in COMMON_ENTITY_PATTERNS:
        # Multiple explicit named humans must not be collapsed into PERSON=1.
        if label == "PERSON" and named:
            continue
        matched_term = next(
            (term for term in sorted(terms, key=len, reverse=True) if re.search(rf"\b{re.escape(term)}s?\b", text)),
            None,
        )
        if not matched_term or matched_term in used_terms:
            continue
        used_terms.add(matched_term)
        count = 1
        count_match = re.search(
            rf"\b(one|two|three|four|five|[1-5])\s+(?:\w+\s+){{0,2}}{re.escape(matched_term)}s?\b",
            text,
        )
        if count_match:
            token = count_match.group(1)
            count = number_words.get(token, int(token) if token.isdigit() else 1)
        locks.append(
            EntityLock(
                label=label,
                expected_count=count,
                description=f"Recurring {matched_term} from the original user request; preserve its established identity exactly.",
            )
        )

    return normalize_entity_locks(locks)[:12]


def _cardinality_block(
    entity_locks: Iterable[EntityLock | dict] | None,
    visible_entity_counts: dict[str, int] | None,
) -> str:
    locks = normalize_entity_locks(entity_locks)
    by_label = {lock.label: lock for lock in locks}
    maxima = ", ".join(f"{lock.label}<={lock.expected_count}" for lock in locks)
    exact_counts: list[str] = []
    for raw_label, raw_count in (visible_entity_counts or {}).items():
        label = _safe_label(raw_label)
        try:
            count = max(0, min(8, int(raw_count)))
        except (TypeError, ValueError):
            continue
        if label in by_label:
            count = min(count, by_label[label].expected_count)
        exact_counts.append(f"{label}={count}")
    parts = []
    if maxima:
        parts.append(f"canonical maxima {maxima}")
    if exact_counts:
        parts.append("THIS SHOT MUST SHOW EXACTLY: " + ", ".join(exact_counts))
    if not parts:
        parts.append("one physical instance of each recurring subject unless explicitly required otherwise")
    return "ENTITY COUNT LOCK: " + "; ".join(parts) + ". Descriptions never instantiate another copy."


def _identity_lock_block(
    entity_locks: Iterable[EntityLock | dict] | None,
    visible_entity_counts: dict[str, int] | None,
    character_bible: str | None,
) -> str:
    locks = normalize_entity_locks(entity_locks)
    visible = {
        _safe_label(label)
        for label, count in (visible_entity_counts or {}).items()
        if int(count or 0) > 0
    }
    selected = [lock for lock in locks if not visible or lock.label in visible][:3]
    lines: list[str] = []
    for lock in selected:
        detail = compact_text(lock.description, 100)
        if detail:
            lines.append(f"{lock.label}: {detail}")
    if not lines and character_bible:
        return "CHARACTER BIBLE / IDENTITY LOCK: " + compact_text(character_bible, 140)
    return "CHARACTER BIBLE / IDENTITY LOCKS: " + " | ".join(lines) if lines else ""


def _scene_body_from_locked_prompt(prompt: str) -> str:
    """Extract only the generation-specific scene body from a planner-locked prompt."""
    matches = list(re.finditer(r"\[SCENE\s+\d+\s+OF\s+\d+\]", prompt, flags=re.IGNORECASE))
    if matches:
        return prompt[matches[-1].end() :].strip()
    match = re.search(r"\[STORY SCENE\]", prompt, flags=re.IGNORECASE)
    if match:
        return prompt[match.end() :].strip()
    return prompt.strip()


def compose_continuity_prompt(
    *,
    scene_prompt: str,
    character_bible: str | None,
    style_bible: str | None,
    scene_index: int | None,
    scene_count: int | None,
    entity_locks: Iterable[EntityLock | dict] | None = None,
    visible_entity_counts: dict[str, int] | None = None,
    reference_frame_present: bool = False,
    prompt_wardrobe_authoritative: bool = False,
    retry_level: int = 0,
    qc_feedback: str | None = None,
) -> str:
    """Build a compact LTX shot prompt; reference frames remain authoritative."""
    prompt = scene_prompt.strip()
    already_locked = f"[{CONTINUITY_HEADER}]" in prompt or f"[{CONTINUITY_HEADER} — CONTINUATION]" in prompt
    if already_locked:
        prompt = _scene_body_from_locked_prompt(prompt)

    scene_label = (
        f"SCENE {scene_index + 1} OF {scene_count}"
        if scene_index is not None and scene_count
        else "STORY SCENE"
    )
    identity = _identity_lock_block(entity_locks, visible_entity_counts, character_bible)
    style = compact_text(style_bible, 110)
    cardinality = _cardinality_block(entity_locks, visible_entity_counts)

    blocks = [f"[{CONTINUITY_HEADER}{' — CONTINUATION' if reference_frame_present else ''}]"]
    if reference_frame_present:
        blocks.append(
            "REFERENCE FRAME IS AUTHORITATIVE: recurring subjects are the one and only canonical physical instance. "
            "Continue existing faces, bodies, props and geography. Preserve wardrobe state unless the current scene explicitly "
            "specifies different clothing; when it does, the current scene wardrobe is authoritative. Describe only the next action/change. "
            "BEGIN MOTION IMMEDIATELY after the anchor frame with natural micro-motion; do not freeze or hold the "
            "opening for several seconds unless the user explicitly requests a still hold. "
            "DO NOT introduce, recreate, re-enter, spawn, mirror or clone them."
        )
    if identity:
        blocks.append(identity)
    if style:
        blocks.append("VISUAL STYLE BIBLE: " + style)
    blocks.append(cardinality)
    blocks.append(
        "HARD CONTINUITY: Never create duplicate copies. No split/fused person, identity swap, body morph, extra face/limb, or unexplained costume/age change."
    )
    if prompt_wardrobe_authoritative:
        blocks.append(
            "WARDROBE OVERRIDE: preserve the recurring person's identity while obeying the CURRENT SCENE clothing description exactly. "
            "Do not pull garments, colors, straps, buttons or patterns from an identity reference unless this scene asks for them."
        )

    if retry_level > 0:
        feedback = compact_text(qc_feedback, 260)
        blocks.append(
            "QC RETRY: prioritize correct identity and subject count over background complexity."
            + (f" Feedback: {feedback}" if feedback else "")
        )

    blocks.extend([f"[{scene_label}]", prompt])
    return "\n\n".join(blocks)



def compose_render_integrity_prompt(
    scene_prompt: str,
    *,
    realism_profile: str = "standard",
    prompt_wardrobe_authoritative: bool = False,
    retry_level: int = 0,
    qc_feedback: str | None = None,
) -> str:
    """Append non-creative render constraints that prevent common AI-video artifacts.

    This is intentionally safe to use when AI prompt enhancement is OFF: it never
    invents story, dialogue, wardrobe or scene content. It only tells the renderer
    not to corrupt the user's already-authored visual requirements.
    """
    prompt = scene_prompt.strip()
    marker = "[TRIVEN VISUAL INTEGRITY]"
    if marker in prompt:
        return prompt

    blocks = [prompt, marker]
    blocks.append(
        "Each subject retains the same recognizable face, anatomy and proportions throughout the shot. "
        "Clothing, props and set geometry remain physically coherent, with stable materials and colors. "
        "Motion follows the described action from the opening frame."
    )
    if prompt_wardrobe_authoritative:
        blocks.append(
            "WARDROBE AUTHORITY: the scene's clothing description overrides clothing in the identity reference."
        )
    if realism_profile in {"real_skin", "identity_max"}:
        blocks.append(
            "Skin and hair retain natural photographic detail under the lighting described in the scene."
        )
    if retry_level > 0 and qc_feedback:
        blocks.append("RENDER QC RETRY: Correct this previously observed issue: " + compact_text(qc_feedback, 300))
    return "\n\n".join(block for block in blocks if block)

def safe_continuity_id(value: str | None) -> str:
    raw = compact_text(value, 96) if value else ""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-")
    return cleaned[:80] or "story"


def local_character_bible(prompt: str) -> str:
    """Deterministic fallback identity lock when Gemini planning is unavailable."""
    named = _named_entity_labels(prompt)
    if named:
        pieces: list[str] = []
        for label in named:
            context = _identity_context(prompt, label, 650)
            pieces.append(
                f"{label}: {context or 'preserve the same named character identity, face, age, body, clothing and recurring props exactly.'}"
            )
        base = " ".join(pieces)
    else:
        base = compact_text(prompt, 1400)
    return (
        "Immutable recurring-character identities. Freeze face geometry, apparent age, hairstyle, hair color, "
        "eye color, skin/fur markings and body proportions. Keep accessories/wardrobe stable unless a scene explicitly "
        "specifies a change; an explicit scene wardrobe overrides earlier clothing while identity remains unchanged. "
        f"{base}"
    )[:2400]


def local_style_bible(prompt: str) -> str:
    base = compact_text(prompt, 1600).lower()
    if any(term in base for term in ("disney", "animated", "animation", "cartoon", "stylized")):
        style = (
            "Polished cinematic feature-animation treatment with expressive but stable character design, "
            "consistent materials, proportions and palette, smooth cinematic camera movement, coherent "
            "environment design, warm story-driven lighting and consistent rendering style from shot to shot."
        )
    elif any(term in base for term in ("photoreal", "realistic", "real human", "human like", "human-like")):
        style = (
            "Photorealistic cinematic treatment with stable human identity, natural skin texture, consistent "
            "hair, realistic fabric/material detail, physically plausible lighting, coherent lens language, "
            "natural depth of field and consistent color science across every shot."
        )
    else:
        style = (
            "Keep the exact same visual language, character rendering, materials, color palette, lighting logic, "
            "lens language and environment design across all scenes."
        )
    return style
