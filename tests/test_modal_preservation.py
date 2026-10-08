import ast
import shutil
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from inference.prompting import PROMPT_FORMAT_VERSION, normalize_ltx_prompt


ROOT = Path(__file__).resolve().parents[1]


class ModalPreservationTests(unittest.TestCase):
    def run_worker(self, failure, *, prompt="presenter"):
        fn = next(node for node in ast.parse((ROOT / "modal/app.py").read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "generate_video")
        fn.decorator_list = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render_prompts = []
            def command(**kwargs):
                return kwargs
            def run(cmd):
                if "input_video_path" in cmd:
                    if failure == "refine":
                        raise RuntimeError("simulated refinement failure")
                    cmd["output_path"].write_bytes(b"refined")
                else:
                    render_prompts.append(cmd["prompt"])
                    cmd["output_path"].write_bytes(b"successful-base-with-audio")
            def mux(**kwargs):
                kwargs["output_path"].write_bytes(b"partial-remux")
                if failure == "mux":
                    raise RuntimeError("simulated remux failure")
                kwargs["output_path"].write_bytes(b"refined-with-original-audio")
                if failure == "missing_base":
                    kwargs["source_video_path"].unlink()
            ns = {"Path": lambda value: root if value == "/tmp" else Path(value), "time": time, "uuid": uuid,
                  "normalize_ltx_prompt": normalize_ltx_prompt, "PROMPT_FORMAT_VERSION": PROMPT_FORMAT_VERSION,
                  "shutil": shutil, "required_model_paths": lambda **kwargs: [], "MODEL_ROOT": root,
                  "build_command": command, "build_refine_details_command": command, "run_ltx_command": run,
                  "preserve_source_audio": mux, "GPU_TYPE": "test GPU", "temporal_chunk_count": lambda value: 1}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal/app.py", "exec"), ns)
            result = ns["generate_video"](prompt, render_mode="dfr", realism_profile="real_skin")
            self.assertEqual(render_prompts, [result["prompt"]])
            return result

    def test_old_api_labels_never_reach_the_worker_render_command(self):
        prompt = '[VISUAL INSTRUCTIONS — NEVER SPOKEN]\nA presenter holds one phone.\n\n[SPOKEN SCRIPT]\nSpeak exactly:\n"Hello."'
        result = self.run_worker(None, prompt=prompt)
        self.assertEqual(result["prompt"], 'A presenter holds one phone.\n\nThe speaker says "Hello.".')
        self.assertEqual(result["prompt_format"], PROMPT_FORMAT_VERSION)

    def test_base_delivered_when_refinement_fails(self):
        result = self.run_worker("refine")
        self.assertEqual(result["video_bytes"], b"successful-base-with-audio")
        self.assertFalse(result["detail_refined"])
        self.assertIn("base render preserved", result["render_details"])

    def test_old_api_audio_repair_labels_do_not_reach_retake_model(self):
        fn = next(node for node in ast.parse((ROOT / "modal/app.py").read_text()).body
                  if isinstance(node, ast.FunctionDef) and node.name == "retake_audio")
        fn.decorator_list = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_prompts = []
            def retake(**kwargs):
                model_prompts.append(kwargs["prompt"])
                kwargs["output_path"].write_bytes(b"repaired-audio-video")
            ns = {"Path": lambda value: root if value == "/tmp" else Path(value), "time": time, "uuid": uuid,
                  "normalize_ltx_prompt": normalize_ltx_prompt, "PROMPT_FORMAT_VERSION": PROMPT_FORMAT_VERSION,
                  "required_model_paths": lambda **kwargs: [], "MODEL_ROOT": root,
                  "retake_audio_only": retake, "GPU_TYPE": "test GPU"}
            exec(compile(ast.Module(body=[fn], type_ignores=[]), "modal/app.py", "exec"), ns)
            result = ns["retake_audio"](b"source", '[SOUND DIRECTION — NEVER SPOKEN]\nWarm voice.\n\n[SPOKEN SCRIPT]\nSpeak exactly:\n"Hello."', 5)
            self.assertEqual(model_prompts, ['Warm voice.\n\nThe speaker says "Hello.".'])
            self.assertEqual(result["prompt_format"], PROMPT_FORMAT_VERSION)
            self.assertEqual(result["video_bytes"], b"repaired-audio-video")

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

    def test_unavailable_base_alternative_does_not_lose_refined_video(self):
        result = self.run_worker("missing_base")
        self.assertEqual(result["video_bytes"], b"refined-with-original-audio")
        self.assertIsNone(result["base_video_bytes"])
        self.assertTrue(result["detail_refined"])


if __name__ == "__main__":
    unittest.main()
