import ast
import shutil
import tempfile
import time
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModalPreservationTests(unittest.TestCase):
    def run_worker(self, failure):
        fn = next(node for node in ast.parse((ROOT / "modal/app.py").read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "generate_video")
        fn.decorator_list = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def command(**kwargs):
                return kwargs
            def run(cmd):
                if "input_video_path" in cmd:
                    if failure == "refine":
                        raise RuntimeError("simulated refinement failure")
                    cmd["output_path"].write_bytes(b"refined")
                else:
                    cmd["output_path"].write_bytes(b"successful-base-with-audio")
            def mux(**kwargs):
                kwargs["output_path"].write_bytes(b"partial-remux")
                if failure == "mux":
                    raise RuntimeError("simulated remux failure")
                kwargs["output_path"].write_bytes(b"refined-with-original-audio")
            ns = {"Path": lambda value: root if value == "/tmp" else Path(value), "time": time, "uuid": uuid,
                  "shutil": shutil, "required_model_paths": lambda **kwargs: [], "MODEL_ROOT": root,
                  "build_command": command, "build_refine_details_command": command, "run_ltx_command": run,
                  "preserve_source_audio": mux, "GPU_TYPE": "test GPU", "temporal_chunk_count": lambda value: 1}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal/app.py", "exec"), ns)
            return ns["generate_video"]("presenter", render_mode="dfr", realism_profile="real_skin")

    def test_base_delivered_when_refinement_fails(self):
        result = self.run_worker("refine")
        self.assertEqual(result["video_bytes"], b"successful-base-with-audio")
        self.assertFalse(result["detail_refined"])
        self.assertIn("base render preserved", result["render_details"])

    def test_base_replaces_partial_remux_when_audio_mux_fails(self):
        result = self.run_worker("mux")
        self.assertEqual(result["video_bytes"], b"successful-base-with-audio")
        self.assertFalse(result["detail_refined"])

    def test_successful_refinement_remains_available(self):
        result = self.run_worker(None)
        self.assertEqual(result["video_bytes"], b"refined-with-original-audio")
        self.assertTrue(result["detail_refined"])
        self.assertEqual(result["base_video_bytes"], b"successful-base-with-audio")
        self.assertIn('base_generation', result['timings_seconds'])
        self.assertIn('detail_refinement', result['timings_seconds'])
        self.assertTrue(all(value >= 0 for value in result['timings_seconds'].values()))


if __name__ == "__main__":
    unittest.main()
