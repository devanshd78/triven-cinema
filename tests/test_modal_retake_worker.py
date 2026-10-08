import importlib.util
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODAL_DIR = ROOT / "modal"
if str(MODAL_DIR) not in sys.path:
    sys.path.append(str(MODAL_DIR))

spec = importlib.util.spec_from_file_location("triven_retake_worker", MODAL_DIR / "retake_worker.py")
assert spec and spec.loader
retake_worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retake_worker)


class ModalRetakeWorkerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required")
    def test_audio_repair_keeps_original_video_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, repaired, output = [Path(tmp) / name for name in ("source.mp4", "repaired.mp4", "output.mp4")]
            for path, color, tone in [(source, "red", "440"), (repaired, "blue", "880")]:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={color}:s=64x64:r=24:d=1", "-f", "lavfi", "-i", f"sine=frequency={tone}:duration=1", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)], check=True, capture_output=True)
            subprocess.run(retake_worker.build_audio_retake_mux_command(source, repaired, output), check=True, capture_output=True)
            def stream_hash(path, stream):
                return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", stream, "-c", "copy", "-f", "hash", "-hash", "sha256", "-"]).strip()
            self.assertEqual(stream_hash(source, "0:v:0"), stream_hash(output, "0:v:0"))
            self.assertEqual(stream_hash(repaired, "0:a:0"), stream_hash(output, "0:a:0"))
            self.assertNotEqual(stream_hash(source, "0:a:0"), stream_hash(output, "0:a:0"))
            self.assertTrue(source.exists())

    def test_retake_command_uses_ltx_virtualenv_python(self):
        command = retake_worker.build_retake_subprocess_command(
            input_path=Path("/tmp/in.mp4"),
            output_path=Path("/tmp/out.mp4"),
            prompt="Keep the video fixed and regenerate natural speech.",
            start_time=0,
            end_time=5,
            seed=7,
        )
        self.assertEqual(command[0], "/opt/LTX-2/.venv/bin/python")
        self.assertTrue(command[1].endswith("retake_worker.py"))
        self.assertIn("--input-path", command)
        self.assertIn("--output-path", command)
        self.assertIn("--prompt", command)
        self.assertIn("--seed", command)

    def test_missing_ltx_virtualenv_has_actionable_error(self):
        with patch.object(retake_worker, "LTX_PYTHON", Path("/definitely/missing/python")):
            with self.assertRaisesRegex(RuntimeError, "Redeploy the current Modal image"):
                retake_worker.retake_audio_only(
                    input_path=Path("/tmp/in.mp4"),
                    output_path=Path("/tmp/out.mp4"),
                    prompt="speech",
                    start_time=0,
                    end_time=1,
                    seed=42,
                )


if __name__ == "__main__":
    unittest.main()
