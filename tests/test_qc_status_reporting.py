"""Verify direct-video QC is visible even without continuity prompting."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.api.routes import generations
from app.schemas.generation import MediaInfo, VideoGenerationRequest
from app.services.continuity_qc import ContinuityQCResult


class DirectQCReportTests(unittest.TestCase):
    def _run_direct(self, qc: ContinuityQCResult, status: str):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            rendered = output / "source.mp4"
            final = output / "delivery.mp4"

            def fake_render(**_kwargs):
                rendered.write_bytes(b"rendered")
                return SimpleNamespace(
                    path=str(rendered),
                    render_seconds=1.5,
                    wall_seconds=1.5,
                    chunk_count=1,
                    gpu="B200",
                    render_details="test",
                    seed=42,
                    provider="modal",
                    detail_refined=False,
                    reference_conditioned=False,
                )

            def fake_delivery(*_args, **_kwargs):
                final.write_bytes(b"delivered")
                return final

            with ExitStack() as stack:
                stack.enter_context(patch.object(settings, "factory_preserve_on_qc_failure", True))
                stack.enter_context(patch.object(settings, "factory_qc_fail_open_on_unavailable", True))
                stack.enter_context(patch.object(generations, "ensure_minimum_free_disk"))
                stack.enter_context(patch.object(generations, "get_video_provider", return_value=SimpleNamespace(name="modal")))
                stack.enter_context(patch.object(generations, "render_long_clip", side_effect=fake_render))
                inspector = stack.enter_context(patch.object(generations, "evaluate_scene_cardinality", return_value=qc))
                stack.enter_context(patch.object(generations, "prepare_delivery", side_effect=fake_delivery))
                stack.enter_context(patch.object(generations, "_media_info", return_value=MediaInfo(
                    width=1280, height=720, has_audio=True, audio_codec="aac", duration_seconds=15.0,
                )))
                stack.enter_context(patch.object(generations, "estimate_gpu_cost", return_value=(0.0, 0.0, "test")))
                stack.enter_context(patch.object(generations, "record_generation_metric"))

                request = VideoGenerationRequest(
                    prompt="A studio presenter reviews a laptop in one shot.",
                    quality="preview", duration_seconds=15,
                    provider="modal", audio_mode="native",
                    continuity_mode="off", continuity_qc_mode="strict",
                    continuity_max_retries=0,
                )
                response = generations._generate_video_impl(request)
                inspector.assert_called_once()
                self.assertTrue(final.exists())
                self.assertEqual(response.visual_qc_status, status)
                self.assertEqual(response.continuity_qc_passed, {
                    "passed": True, "failed": False, "unavailable": None,
                }[status])
                if status != "passed":
                    self.assertTrue(response.continuity_warnings)

    def test_direct_visual_qc_passed(self):
        self._run_direct(ContinuityQCResult(passed=True), "passed")

    def test_direct_visual_qc_failed_retains_video(self):
        self._run_direct(ContinuityQCResult(passed=False, artifact_detected=True, note="Frame contains split-screen artifact."), "failed")

    def test_direct_visual_qc_unavailable_retains_video(self):
        self._run_direct(ContinuityQCResult(passed=True, skipped=True, note="Gemini HTTP 429"), "unavailable")
