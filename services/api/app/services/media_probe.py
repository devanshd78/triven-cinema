import json
import subprocess

from app.core.config import settings
from pathlib import Path


class MediaProbeError(RuntimeError):
    pass


def probe_media(path: Path, *, include_concat_signature: bool = False) -> dict:
    if not path.exists():
        raise MediaProbeError(f"Media file does not exist: {path}")

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
    ]
    if include_concat_signature:
        command.extend(["-show_data_hash", "sha256"])
    command.append(str(path))
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(5, int(settings.ffprobe_timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError("ffprobe timed out while inspecting media.") from exc
    if process.returncode != 0:
        raise MediaProbeError(
            "ffprobe failed:\n" + process.stderr[-5000:]
        )

    try:
        payload = json.loads(process.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid JSON.") from exc

    streams = payload.get("streams") or []
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )

    duration = None
    format_duration = (payload.get("format") or {}).get("duration")
    if format_duration is not None:
        try:
            duration = round(float(format_duration), 3)
        except (TypeError, ValueError):
            duration = None

    result = {
        "width": int(video_stream.get("width")) if video_stream and video_stream.get("width") else None,
        "height": int(video_stream.get("height")) if video_stream and video_stream.get("height") else None,
        "duration_seconds": duration,
        "video_codec": video_stream.get("codec_name") if video_stream else None,
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "audio_channels": int(audio_stream.get("channels")) if audio_stream and audio_stream.get("channels") else None,
        "format_name": (payload.get("format") or {}).get("format_name"),
        "size_bytes": path.stat().st_size,
    }
    if include_concat_signature:
        # Concat's stream-copy path requires identical stream layouts, timing,
        # and codec headers. Missing metadata or unusual streams use encoding.
        fields = (
            "codec_type", "codec_name", "codec_tag_string", "profile", "level",
            "time_base", "extradata_hash", "width", "height", "pix_fmt",
            "field_order", "sample_aspect_ratio", "color_range", "color_space",
            "color_transfer", "color_primaries", "chroma_location", "has_b_frames",
            "r_frame_rate", "avg_frame_rate", "sample_fmt", "sample_rate",
            "channels", "channel_layout",
        )
        supported = (
            [stream.get("codec_type") for stream in streams] in (["video"], ["video", "audio"])
            and video_stream.get("codec_name") == "h264"
            and video_stream.get("pix_fmt") == "yuv420p"
            and all(video_stream.get(key) for key in ("width", "height", "time_base", "r_frame_rate", "extradata_hash"))
            and not any(stream.get("side_data_list") for stream in streams)
            and (audio_stream is None or (
                audio_stream.get("codec_name") == "aac"
                and all(audio_stream.get(key) for key in ("time_base", "sample_rate", "channels", "extradata_hash"))
            ))
        )
        result["concat_signature"] = (
            [{field: stream.get(field) for field in fields} for stream in streams]
            if supported else None
        )
    return result
