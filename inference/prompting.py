"""Compile scene descriptions for LTX, without application instruction headings."""
import json
import re


PROMPT_FORMAT_VERSION = "ltx-native-scene-v1"

_VISUAL_HEADER = "[VISUAL INSTRUCTIONS — NEVER SPOKEN]"
_SOUND_HEADER = "[SOUND DIRECTION — NEVER SPOKEN]"
_SCRIPT_HEADER = "[SPOKEN SCRIPT]"
_APP_MARKERS = re.compile(
    r"(?m)^\s*\[(?:TRIVEN VISUAL INTEGRITY|TRIVEN CONTINUITY LOCK(?: — CONTINUATION)?|"
    r"SCENE \d+ OF \d+|STORY SCENE)\]\s*$"
)
_APP_LABELS = re.compile(
    r"(?m)^(?:CHARACTER BIBLE / IDENTITY LOCKS?|VISUAL STYLE BIBLE|HARD CONTINUITY|"
    r"WARDROBE OVERRIDE|WARDROBE AUTHORITY|RENDER QC RETRY|QC RETRY|START FRAME BEHAVIOR):\s*"
)


def _scene_text(value: str) -> str:
    value = _APP_MARKERS.sub("", value.strip())
    value = _APP_LABELS.sub("", value)
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def compile_native_prompt(visual: str, script: str, sound: str | None = None) -> str:
    """Keep scene, sound and quoted speech in the model's descriptive format.

    Ingredients' trained Reference sheet / Generated video labels are retained.
    Application-only labels such as NEVER SPOKEN must not become visual tokens.
    The immutable script remains separately stored and checked by audio QC.
    """
    blocks = [_scene_text(visual)] if visual.strip() else []
    if sound and sound.strip():
        blocks.append(sound.strip())
    if script.strip():
        blocks.append("The speaker says " + json.dumps(script.strip(), ensure_ascii=False) + ".")
    else:
        blocks.append("The scene contains the requested ambient sound, without speech.")
    return "\n\n".join(blocks)


def normalize_ltx_prompt(prompt: str) -> str:
    """Adapt requests from older deployed APIs at the worker boundary, too.

    Only our exact legacy envelope is parsed. User headings and quoted on-screen
    text are left intact. Malformed legacy speech must not be silently dropped.
    """
    prompt = prompt.strip()
    if not prompt.startswith((_VISUAL_HEADER, _SOUND_HEADER, _SCRIPT_HEADER)):
        return _scene_text(prompt)
    if _SCRIPT_HEADER not in prompt:
        raise ValueError("Legacy generation prompt is missing its speech section; update the API before retrying.")
    before, script_block = prompt.rsplit(_SCRIPT_HEADER, 1)
    visual = before.removeprefix(_VISUAL_HEADER).strip()
    sound = ""
    if _SOUND_HEADER in visual:
        visual, sound = visual.split(_SOUND_HEADER, 1)
    script_block = script_block.strip()
    if script_block.startswith("No dialogue, narration, singing"):
        script = ""
    else:
        try:
            script = json.loads(script_block.rsplit("\n", 1)[-1])
        except (ValueError, TypeError) as exc:
            raise ValueError("Legacy spoken dialogue could not be preserved; update the API before retrying.") from exc
        if not isinstance(script, str):
            raise ValueError("Legacy spoken dialogue must be text.")
    return compile_native_prompt(visual.strip(), script, sound.strip())
