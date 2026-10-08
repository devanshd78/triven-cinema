"""Regression tests: expensive rendered clips must remain reviewable after QC rejection."""
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.core.config import settings
from app.schemas.factory import FactoryGenerationRequest
from app.services import factory_service
from app.services.continuity_qc import ContinuityQCResult


class RetainQCRejectedVideoTests(unittest.TestCase):
    def _run_fake_factory(self, *, preserve: bool, publish: bool = False,
                          qc_override: ContinuityQCResult | None = None,
                          expected_status: str = "failed") -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            scene_video = output_dir / "scene.mp4"
            final_video = output_dir / "final.mp4"
            prompt = "A presenter explains a product in a studio."
            plan = SimpleNamespace(
                scenes=[SimpleNamespace(prompt=prompt, visible_entity_counts={})],
                source="prompt_only",
                note="test",
                character_bible="",
                style_bible="",
                entity_locks=[],
            )
            provider = SimpleNamespace(name="modal", supports_audio_retake=False)

            def fake_render(**_kwargs):
                scene_video.write_bytes(b"LTX_RENDERED_VIDEO")
                return SimpleNamespace(
                    path=str(scene_video),
                    render_seconds=1.0,
                    wall_seconds=1.0,
                    chunk_count=1,
                    gpu="B200",
                    detail_refined=False,
                    render_details="simulated LTX success",
                )

            def fake_combine(_sources, target):
                target.write_bytes(b"COMPOSED_VIDEO")

            def fake_delivery(*_args, **_kwargs):
                final_video.write_bytes(b"FINAL_VIDEO")
                return final_video

            qc = qc_override or ContinuityQCResult(
                passed=False,
                duplicate_detected=True,
                identity_drift_detected=True,
                note="Generated scene contains a split-screen and identity drift.",
                confidence=0.95,
            )
            with ExitStack() as stack:
                stack.enter_context(patch.object(settings, "factory_preserve_on_qc_failure", preserve))
                stack.enter_context(patch.object(settings, "factory_qc_fail_open_on_unavailable", True))
                stack.enter_context(patch.object(factory_service, "ensure_minimum_free_disk"))
                stack.enter_context(patch.object(factory_service, "resolve_element_bindings", return_value=[]))
                stack.enter_context(patch.object(factory_service, "create_prompt_only_plan", return_value=plan))
                stack.enter_context(patch.object(factory_service, "get_video_provider", return_value=provider))
                stack.enter_context(patch.object(factory_service, "render_long_clip", side_effect=fake_render))
                stack.enter_context(patch.object(factory_service, "evaluate_scene_cardinality", return_value=qc))
                stack.enter_context(patch.object(factory_service, "combine_videos", side_effect=fake_combine))
                stack.enter_context(patch.object(factory_service, "prepare_delivery", side_effect=fake_delivery))
                stack.enter_context(patch.object(factory_service, "_media_info", return_value=SimpleNamespace(
                    duration_seconds=15.0, has_audio=False, width=1280, height=720,
                )))
                stack.enter_context(patch.object(factory_service, "estimate_gpu_cost", return_value=(0.0, 0.0, "test")))
                stack.enter_context(patch.object(factory_service, "record_generation_metric"))
                uploader = stack.enter_context(patch.object(factory_service, "upload_video"))

                request = FactoryGenerationRequest(
                    prompt=prompt,
                    target_duration_seconds=15,
                    scene_duration_seconds=15,
                    quality="preview",
                    audio_mode="mute",
                    continuity_mode="strict",
                    continuity_qc_mode="strict",
                    continuity_max_retries=0,
                    publish_to_youtube=publish,
                    youtube_title="Test presenter video" if publish else None,
                    enhance_prompt=False,
                )
                result = factory_service.run_factory_generation(request, workspace_id="demo")
                self.assertTrue(scene_video.exists(), "QC must not delete the rendered scene")
                self.assertEqual(final_video.read_bytes(), b"FINAL_VIDEO")
                self.assertEqual(result.final_filename, final_video.name)
                self.assertEqual(result.visual_qc_status, expected_status)
                self.assertEqual(result.audio_qc_status, "not_checked")
                if expected_status == "failed":
                    self.assertFalse(result.continuity_qc_passed)
                    self.assertTrue(any("retained for review" in note for note in result.continuity_warnings))
                    self.assertTrue(any("split-screen" in note for note in result.continuity_warnings))
                elif expected_status == "unavailable":
                    self.assertIsNone(result.continuity_qc_passed)
                    self.assertTrue(any("Gemini" in note for note in result.continuity_warnings))
                else:
                    self.assertTrue(result.continuity_qc_passed)
                if publish:
                    uploader.assert_not_called()
                    self.assertIsNone(result.youtube_url)
                    self.assertTrue(any("YouTube publishing skipped" in note for note in result.continuity_warnings))

    def test_strict_qc_failure_keeps_reviewable_video(self):
        self._run_fake_factory(preserve=True)

    def test_strict_qc_failure_skips_auto_publish(self):
        self._run_fake_factory(preserve=True, publish=True)

    def test_legacy_flag_cannot_destroy_render(self):
        self._run_fake_factory(preserve=False)

    def test_qc_pass_is_reported(self):
        self._run_fake_factory(preserve=True, qc_override=ContinuityQCResult(passed=True), expected_status="passed")

    def test_unavailable_qc_is_explicit_and_prevents_auto_publish(self):
        self._run_fake_factory(
            preserve=True,
            publish=True,
            qc_override=ContinuityQCResult(passed=True, skipped=True, note="Gemini HTTP 429"),
            expected_status="unavailable",
        )
