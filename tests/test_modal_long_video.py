import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODAL_DIR = ROOT / "modal"
if str(MODAL_DIR) not in sys.path:
    sys.path.append(str(MODAL_DIR))

spec = importlib.util.spec_from_file_location("triven_ltx_worker_long", MODAL_DIR / "ltx_worker.py")
assert spec and spec.loader
ltx_worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ltx_worker)


class ModalLongVideoTests(unittest.TestCase):
    def test_generation_and_refinement_use_built_environment_without_uv_sync(self):
        generation = ltx_worker.build_command(prompt='presenter', output_path=Path('/tmp/out.mp4'),
                                             width=1024, height=576, duration_seconds=5, seed=42, decoder='conv')
        refinement = ltx_worker.build_refine_details_command(input_video_path=Path('/tmp/in.mp4'),
                output_path=Path('/tmp/out.mp4'), width=1024, height=576, duration_seconds=5, seed=42)
        for command in (generation, refinement):
            self.assertEqual(command[:2], ['/opt/LTX-2/.venv/bin/python', '-m'])
            self.assertNotIn('uv', command)
            self.assertEqual(command[command.index('--quantization') + 1], 'fp8-cast')

    def test_thirty_second_clip_uses_native_temporal_windowing(self):
        command = ltx_worker.build_command(
            prompt="A continuous cinematic tracking shot with synchronized natural audio.",
            output_path=Path("/tmp/out.mp4"),
            width=1024,
            height=576,
            duration_seconds=30,
            seed=42,
            decoder="conv",
        )
        self.assertIn("--chunk-pixel-frames", command)
        self.assertIn("--chunk-carry-frames", command)
        self.assertNotIn("--chunk-blend-frames", command)
        self.assertGreater(ltx_worker.temporal_chunk_count(30), 1)
        self.assertEqual(ltx_worker.LONG_VIDEO_PIXEL_FRAMES, 97)

    def test_twenty_second_distilled_clip_remains_single_window(self):
        command = ltx_worker.build_command(
            prompt="A twenty second continuous cinematic shot.",
            output_path=Path("/tmp/out.mp4"),
            width=1024,
            height=576,
            duration_seconds=20,
            seed=42,
            decoder="conv",
        )
        self.assertNotIn("--chunk-pixel-frames", command)
        self.assertEqual(ltx_worker.temporal_chunk_count(20), 1)
        frames_index = command.index("--num-frames") + 1
        self.assertEqual(command[frames_index], "481")

    def test_dfr_uses_production_pipeline_detailing_lora_and_diffusion_vae(self):
        command = ltx_worker.build_command(
            prompt="Radha hears the flute beside the Yamuna.",
            output_path=Path("/tmp/out.mp4"),
            width=1024,
            height=576,
            duration_seconds=8,
            seed=42,
            decoder="conv",
            render_mode="dfr",
        )
        joined = " ".join(command)
        self.assertIn("ltx_pipelines.dfr_pipeline", command)
        self.assertIn("--detailing-lora", command)
        self.assertIn("ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors", joined)
        self.assertIn("ltx-2.5-video-vae-bf16.safetensors", joined)
        self.assertNotIn("--chunk-pixel-frames", command)

    def test_dfr_thirty_second_1080p_is_one_pipeline_call_without_chunk_flags(self):
        command = ltx_worker.build_command(
            prompt="One continuous thirty second cinematic shot with no artificial chunk cuts.",
            output_path=Path("/tmp/out.mp4"),
            width=1920,
            height=1088,
            duration_seconds=30,
            seed=42,
            decoder="diffusion",
            render_mode="dfr",
        )
        self.assertIn("ltx_pipelines.dfr_pipeline", command)
        self.assertNotIn("--chunk-pixel-frames", command)
        self.assertNotIn("--chunk-carry-frames", command)
        frames_index = command.index("--num-frames") + 1
        self.assertEqual(command[frames_index], "721")

    def test_ingredients_uses_ic_lora_and_can_compose_previous_frame(self):
        command = ltx_worker.build_command(
            prompt="Reference sheet: Radha and Krishna. Generated video: Radha approaches Krishna.",
            output_path=Path("/tmp/out.mp4"),
            width=1024,
            height=576,
            duration_seconds=15,
            seed=42,
            decoder="conv",
            render_mode="dfr",
            reference_image_path=Path("/tmp/previous.png"),
            reference_strength=0.85,
            element_reference_video_path=Path("/tmp/reference.mp4"),
            element_reference_strength=1.0,
        )
        joined = " ".join(command)
        self.assertIn("ltx_pipelines.ic_lora", command)
        self.assertIn("--video-conditioning", command)
        self.assertIn("--lora", command)
        self.assertIn("ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors", joined)
        self.assertIn("--image", command)
        self.assertIn("--stage-2-ic-lora", command)
        self.assertIn("--chunk-pixel-frames", command)
        self.assertEqual(command[command.index("--chunk-pixel-frames") + 1], "121")
        self.assertEqual(command[command.index("--chunk-carry-frames") + 1], "25")
        self.assertGreater(ltx_worker.ingredients_temporal_chunk_count(15), 1)
        self.assertNotIn("--detailing-lora", command)

    def test_ingredients_accepts_thirty_second_identity_scene_with_temporal_windows_and_stage2_lock(self):
        command = ltx_worker.build_command(
            prompt="Reference sheet: Radha. Generated video: one long shot.",
            output_path=Path("/tmp/out.mp4"),
            width=1920,
            height=1088,
            duration_seconds=30,
            seed=42,
            decoder="diffusion",
            render_mode="dfr",
            element_reference_video_path=Path("/tmp/reference.mp4"),
        )
        self.assertIn("ltx_pipelines.ic_lora", command)
        self.assertIn("--stage-2-ic-lora", command)
        self.assertIn("--chunk-pixel-frames", command)
        self.assertGreater(ltx_worker.ingredients_temporal_chunk_count(30), 1)
        self.assertIn("ltx-2.5-video-vae-bf16.safetensors", " ".join(command))

    def test_ingredients_large_canvas_uses_spatial_tiling_with_stage2_lock(self):
        command = ltx_worker.build_command(
            prompt="Reference sheet: presenter. Generated video: stable presenter.",
            output_path=Path("/tmp/out.mp4"),
            width=3840,
            height=2176,
            duration_seconds=15,
            seed=42,
            decoder="diffusion",
            render_mode="dfr",
            element_reference_video_path=Path("/tmp/reference.mp4"),
        )
        self.assertIn("--tile", command)
        self.assertIn("--stage-2-ic-lora", command)
        self.assertEqual(command[command.index("--tile-height") + 1], "576")
        self.assertEqual(command[command.index("--tile-width") + 1], "1024")


    def test_ingredients_reference_video_matches_target_length_and_resolution(self):
        frames = ltx_worker.static_reference_frame_count(15)
        self.assertEqual(frames, 361)
        command = ltx_worker.build_static_reference_video_command(
            Path("/tmp/sheet.png"),
            Path("/tmp/reference.mp4"),
            width=1920,
            height=1088,
            frame_count=frames,
        )
        self.assertNotIn("-t", command)
        frames_index = command.index("-frames:v") + 1
        self.assertEqual(command[frames_index], "361")
        vf = command[command.index("-vf") + 1]
        self.assertIn("scale=1920:1088", vf)
        self.assertIn("pad=1920:1088", vf)
        self.assertEqual(command[command.index("-r") + 1], "24")

    def test_ingredients_reference_video_never_drops_below_training_bucket(self):
        self.assertEqual(ltx_worker.static_reference_frame_count(1), 121)
        self.assertEqual(ltx_worker.static_reference_frame_count(5), 121)


    def test_refine_details_uses_official_tiled_ic_lora_and_diffusion_vae(self):
        command = ltx_worker.build_refine_details_command(
            input_video_path=Path("/tmp/base.mp4"),
            output_path=Path("/tmp/refined.mp4"),
            width=1920,
            height=1088,
            duration_seconds=8,
            seed=99,
        )
        joined = " ".join(command)
        self.assertIn("ltx_pipelines.ic_lora", command)
        self.assertIn("ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors", joined)
        self.assertIn("ltx-2.5-video-vae-bf16.safetensors", joined)
        self.assertIn("--tile", command)
        self.assertIn("--stage-2-ic-lora", command)
        self.assertEqual(command[command.index("--tile-height") + 1], "576")
        self.assertEqual(command[command.index("--tile-width") + 1], "1024")
        self.assertNotIn("--chunk-pixel-frames", command)

    def test_refine_details_streams_long_scene_in_97_frame_windows(self):
        command = ltx_worker.build_refine_details_command(
            input_video_path=Path("/tmp/base.mp4"),
            output_path=Path("/tmp/refined.mp4"),
            width=1920,
            height=1088,
            duration_seconds=15,
            seed=99,
        )
        self.assertIn("--chunk-pixel-frames", command)
        self.assertEqual(command[command.index("--chunk-pixel-frames") + 1], "97")
        self.assertIn("--chunk-carry-frames", command)

    def test_refine_details_keeps_source_audio_in_final_mux(self):
        command = ltx_worker.build_preserve_source_audio_command(
            refined_video_path=Path("/tmp/refined.mp4"),
            source_video_path=Path("/tmp/base.mp4"),
            output_path=Path("/tmp/final.mp4"),
        )
        joined = " ".join(command)
        self.assertIn("-map 0:v:0", joined)
        self.assertIn("-map 1:a:0?", joined)
        self.assertIn("-c:a copy", joined)

    def test_dfr_rejects_more_than_thirty_seconds_single_pass(self):
        with self.assertRaises(ValueError):
            ltx_worker.build_command(
                prompt="An excessively long single-pass shot.",
                output_path=Path("/tmp/out.mp4"),
                width=1920,
                height=1088,
                duration_seconds=31,
                seed=42,
                decoder="diffusion",
                render_mode="dfr",
            )


if __name__ == "__main__":
    unittest.main()
