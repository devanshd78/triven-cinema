"""Exercise composition with real encoded media, including the safe fallback."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import video_combiner
from app.services.media_probe import probe_media


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VideoCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        for name, color, fps, audio in (
            ("first", "red", 24, True), ("second", "blue", 24, True),
            ("silent", "green", 24, False), ("different_fps", "yellow", 30, True),
        ):
            command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                       f"color=c={color}:s=160x96:r={fps}:d=0.5"]
            if audio:
                command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.5"]
            command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
            if audio:
                command += ["-c:a", "aac", "-b:a", "192k"]
            command += ["-movflags", "+faststart", str(cls.root / f"{name}.mp4")]
            subprocess.run(command, check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def frame_hashes(self, path):
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
             "-f", "framemd5", "-"], check=True, capture_output=True, text=True, timeout=30,
        )
        return [line.rsplit(",", 1)[-1].strip() for line in result.stdout.splitlines()
                if line and not line.startswith("#")]

    def test_single_clip_is_preserved_byte_for_byte(self):
        source = self.root / "first.mp4"
        output = self.root / "single-output.mp4"
        with patch.object(video_combiner, "_run_ffmpeg") as run:
            self.assertEqual(video_combiner.combine_videos([source], output), output)
        run.assert_not_called()
        self.assertEqual(source.read_bytes(), output.read_bytes())

    def test_single_clip_can_already_be_at_destination(self):
        source = self.root / "first.mp4"
        original = source.read_bytes()
        self.assertEqual(video_combiner.combine_videos([source], source), source)
        self.assertEqual(source.read_bytes(), original)

    def test_compatible_clips_keep_every_picture_and_audio(self):
        sources = [self.root / "first.mp4", self.root / "second.mp4"]
        output = self.root / "joined.mp4"
        with patch.object(video_combiner, "_run_ffmpeg", wraps=video_combiner._run_ffmpeg) as run:
            video_combiner.combine_videos(sources, output)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertEqual(self.frame_hashes(output), sum((self.frame_hashes(path) for path in sources), []))
        info = probe_media(output)
        self.assertTrue(info["has_audio"])
        self.assertAlmostEqual(info["duration_seconds"], 1.0, delta=0.1)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                       check=True, capture_output=True, timeout=30)

    def test_matching_silent_clips_use_stream_copy(self):
        source = self.root / "silent.mp4"
        output = self.root / "silent-joined.mp4"
        with patch.object(video_combiner, "_run_ffmpeg", wraps=video_combiner._run_ffmpeg) as run:
            video_combiner.combine_videos([source, source], output)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertFalse(probe_media(output)["has_audio"])
        self.assertEqual(self.frame_hashes(output), self.frame_hashes(source) * 2)

    def test_different_frame_rates_use_encoding(self):
        sources = [self.root / "first.mp4", self.root / "different_fps.mp4"]
        signatures = [probe_media(path, include_concat_signature=True)["concat_signature"] for path in sources]
        self.assertNotEqual(signatures[0], signatures[1])
        output = self.root / "different-fps-joined.mp4"
        with patch.object(video_combiner, "_run_ffmpeg", wraps=video_combiner._run_ffmpeg) as run:
            video_combiner.combine_videos(sources, output)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertTrue(probe_media(output)["has_audio"])

    def test_mixed_audio_keeps_normalization_and_encoding(self):
        output = self.root / "mixed-joined.mp4"
        with patch.object(video_combiner, "_run_ffmpeg", wraps=video_combiner._run_ffmpeg) as run:
            video_combiner.combine_videos([self.root / "first.mp4", self.root / "silent.mp4"], output)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        info = probe_media(output)
        self.assertTrue(info["has_audio"])
        self.assertAlmostEqual(info["duration_seconds"], 1.0, delta=0.1)
        self.assertEqual(list(self.root.glob(".normalized-*")), [])

    def test_failed_stream_copy_falls_back_without_touching_sources(self):
        sources = [self.root / "first.mp4", self.root / "second.mp4"]
        originals = [path.read_bytes() for path in sources]
        output = self.root / "fallback-joined.mp4"
        real_run = video_combiner._run_ffmpeg
        commands = []

        def fail_copy(command):
            commands.append(command)
            if command[command.index("-c:v") + 1] == "copy":
                Path(command[-1]).write_bytes(b"partial failed output")
                raise video_combiner.VideoCombineError("Simulated unsupported stream copy")
            real_run(command)

        with patch.object(video_combiner, "_run_ffmpeg", side_effect=fail_copy):
            video_combiner.combine_videos(sources, output)
        self.assertEqual(len(commands), 2)
        self.assertEqual([path.read_bytes() for path in sources], originals)
        self.assertEqual(len(self.frame_hashes(output)), 24)
        self.assertTrue(probe_media(output)["has_audio"])
        self.assertEqual(list(self.root.glob("*.txt")), [])
