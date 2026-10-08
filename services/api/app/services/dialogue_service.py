"""Keep user-authored speech outside lossy visual-prompt planning."""

import json
import re


_QUOTED = r'(?P<quote>"[^"\n]+"|“[^”\n]+”|‘[^’\n]+’|\x27[^\x27\n]+\x27)'
_SPOKEN_QUOTE = re.compile(
    r'\b(?:says?|speaks?|narrates?|whispers?|replies|asks?|dialogue|narration)\b'
    r'[^\n.!?"“‘\x27]{0,50}?' + _QUOTED, re.IGNORECASE,
)
_HEADING_QUOTE = re.compile(r'(?m)^#{1,6}[^\n]+\n\s*' + _QUOTED)


def _speech_matches(prompt: str) -> list[re.Match]:
    matches = sorted([*_SPOKEN_QUOTE.finditer(prompt), *_HEADING_QUOTE.finditer(prompt)], key=lambda item: item.start())
    selected = []
    for match in matches:
        if not selected or match.start() >= selected[-1].end():
            selected.append(match)
    return selected


def extract_spoken_script(prompt: str) -> str:
    """Conservatively migrate explicitly spoken quotes from legacy prompts.

    Other quoted visual terms are not treated as dialogue. The dedicated script
    field is authoritative whenever supplied and supports arbitrary languages.
    """
    return "\n".join(match.group("quote")[1:-1] for match in _speech_matches(prompt or ""))


def separate_visual_and_script(prompt: str, spoken_script: str = "") -> tuple[str, str]:
    matches = _speech_matches(prompt or "")
    script = spoken_script.strip() or "\n".join(match.group("quote")[1:-1] for match in matches)
    visual = prompt or ""
    for match in reversed(matches):
        # Remove the spoken quote, retaining character names and scene headings.
        start, end = match.span("quote")
        visual = visual[:start] + visual[end:]
    if spoken_script.strip():
        # Explicit scripts can also appear verbatim in older combined prompts.
        visual = visual.replace(spoken_script.strip(), "")
    return visual.strip(), script


def split_spoken_script(script: str, scene_count: int) -> list[str]:
    """Assign each source sentence once; never repeat speech to fill runtime."""
    count = max(1, scene_count)
    text = (script or "").strip()
    if not text:
        return [""] * count
    if count == 1:
        return [text]
    sentences = [item.strip() for item in re.split(r'(?<=[.!?。！？])\s+|\n+', text) if item.strip()]
    # Keep short utterances intact. A long unpunctuated narration can be divided
    # at whitespace without dropping or rewriting any source words.
    if len(sentences) == 1 and len(text.split()) > 40:
        words = text.split()
        return [" ".join(words[index * len(words) // count:(index + 1) * len(words) // count]) for index in range(count)]
    output = [""] * count
    assigned = min(count, len(sentences))
    for index in range(assigned):
        start = index * len(sentences) // assigned
        end = (index + 1) * len(sentences) // assigned
        output[index] = " ".join(sentences[start:end])
    return output


def build_audio_prompt(visual_prompt: str, spoken_script: str, audio_direction: str | None = None) -> str:
    """Compile LTX's single text input from separately stored visual/audio data."""
    visual, _ = separate_visual_and_script(visual_prompt, spoken_script)
    blocks = []
    if visual:
        blocks.append("[VISUAL INSTRUCTIONS — NEVER SPOKEN]\n" + visual)
    if audio_direction and audio_direction.strip():
        blocks.append("[SOUND DIRECTION — NEVER SPOKEN]\n" + audio_direction.strip())
    script = (spoken_script or "").strip()
    blocks.append(
        "[SPOKEN SCRIPT]\nSpeak exactly the following words once, in order; no other narration, "
        "instructions, repetition or invented speech. Synchronize the visible speaker's lips:\n"
        + json.dumps(script, ensure_ascii=False)
        if script else
        "[SPOKEN SCRIPT]\nNo dialogue, narration, singing, chanting or speech-like vocal sounds. Use only requested ambience, Foley or music."
    )
    return "\n\n".join(blocks)
