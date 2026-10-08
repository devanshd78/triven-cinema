"""No provider calls: paid render recovery and QC authorization regressions."""
import json
import tempfile
import time
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.core.config import settings
from app.schemas.factory import FactoryGenerationRequest
from app.schemas.generation import MediaInfo, VideoGenerationRequest
from app.services import factory_service as factory
from app.services import render_review_service as review
from app.services import continuity_qc, audio_qc
from app.services.continuity_qc import ContinuityQCResult
from app.services.audio_qc import AudioQCResult
from app.services.job_service import get_job, submit_job, workspace_owns_generated_file
from app.api.routes import generations
from inference.providers.base import VideoGenerationResult


def refined_take(root):
    refined = root / "refined.mp4"
    base = root / "before-refinement.mp4"
    refined.write_bytes(b"bad texture pass")
    base.write_bytes(b"original picture and audio")
    return VideoGenerationResult(
        filename=refined.name, path=str(refined), base_path=str(base), seed=42,
        render_details="Test texture pass", render_seconds=120, wall_seconds=125,
        prompt="A presenter in a studio.", provider="modal", detail_refined=True,
    )


@contextmanager
def fake_factory(*, fail_render=None, visual=None, audio=None, fail_retake=False,
                 fail_delivery=False, fail_publish=False, count=1):
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp).resolve()
        state = SimpleNamespace(paths=[], render_calls=0)
        script = "Welcome to Triven Cinema."
        plan = SimpleNamespace(scenes=[SimpleNamespace(prompt="A presenter in a studio.", spoken_script=script if i == 0 else "",
                                                       visible_entity_counts={}) for i in range(count)],
                               source="direct", note="test", character_bible="", style_bible="", entity_locks=[])
        provider = SimpleNamespace(name="modal", supports_audio_retake=fail_retake,
                                   retake_audio=Mock(side_effect=RuntimeError("repair unavailable")))
        def render(**kwargs):
            state.render_calls += 1
            if state.render_calls == fail_render:
                raise RuntimeError("GPU failed")
            path = root / f"{root.name}-source-{state.render_calls}.mp4"
            path.write_bytes(b"rendered")
            state.paths.append(path)
            return SimpleNamespace(path=str(path), render_seconds=1., wall_seconds=1., chunk_count=1,
                                   gpu="B200", detail_refined=False, render_details="test", seed=42,
                                   provider="modal", reference_conditioned=False)
        def combine(paths, target):
            target.write_bytes(b"combined")
        def delivery(source, **kwargs):
            if fail_delivery:
                raise RuntimeError("ffmpeg unavailable")
            path = root / f"{root.name}-delivery.mp4"
            path.write_bytes(b"delivery")
            return path
        state.renderer = Mock(side_effect=render)
        state.inspector = Mock(return_value=visual or ContinuityQCResult(passed=True, note="Passed"))
        state.publisher = Mock(side_effect=RuntimeError("upload unavailable") if fail_publish else None,
                               return_value={"video_id": "test", "youtube_url": "https://example.invalid", "privacy": "private"})
        for name, value in {
            "GENERATED_DIR": root, "ensure_minimum_free_disk": Mock(), "resolve_element_bindings": Mock(return_value=[]),
            "create_prompt_only_plan": Mock(return_value=plan), "get_video_provider": Mock(return_value=provider),
            "render_long_clip": state.renderer, "evaluate_scene_cardinality": state.inspector,
            "combine_videos": Mock(side_effect=combine), "prepare_delivery": Mock(side_effect=delivery),
            "_media_info": Mock(return_value=MediaInfo(duration_seconds=count * 15, has_audio=True, width=1280, height=720)),
            "estimate_gpu_cost": Mock(return_value=(0., 0., "test")), "record_generation_metric": Mock(),
            "upload_video": state.publisher,
        }.items():
            stack.enter_context(patch.object(factory, name, value))
        stack.enter_context(patch.object(review, "evaluate_scene_audio", return_value=audio or AudioQCResult(passed=True, note="Passed")))
        stack.enter_context(patch.object(settings, "factory_audio_qc_enabled", True))
        stack.enter_context(patch.object(settings, "factory_audio_retake_enabled", True))
        stack.enter_context(patch.object(settings, "factory_preserve_on_qc_failure", True))
        stack.enter_context(patch.object(settings, "factory_qc_fail_open_on_unavailable", True))
        state.root, state.provider = root, provider
        yield state


def request(**changes):
    return FactoryGenerationRequest(prompt="A presenter in a studio.", spoken_script="Welcome to Triven Cinema.",
                                    target_duration_seconds=changes.pop("target_duration_seconds", 15),
                                    scene_duration_seconds=15, quality="preview", enhance_prompt=False,
                                    audio_mode=changes.pop("audio_mode", "mute"), **changes)


class RenderRecoveryTests(unittest.TestCase):
    def test_good_base_avoids_another_factory_gpu_render(self):
        with fake_factory() as state:
            take = refined_take(state.root)
            state.renderer.side_effect = None
            state.renderer.return_value = take
            state.inspector.side_effect = [ContinuityQCResult(passed=False, artifact_detected=True, note="Glare overlay"),
                                           ContinuityQCResult(passed=True, note="Clean studio picture")]
            result = factory.run_factory_generation(request(continuity_max_retries=1), workspace_id="audit")
            state.renderer.assert_called_once()
            self.assertEqual(result.visual_qc_status, "passed")
            self.assertEqual(result.continuity_regenerations, 0)
            self.assertFalse(result.detail_refined)
            self.assertEqual(result.scene_results[0]["filename"], Path(take.base_path).name)
            self.assertEqual(len(result.scene_results[0]["visual_qc_attempts"]), 2)
            self.assertTrue(Path(take.path).exists())
            self.assertTrue(workspace_owns_generated_file("audit", Path(take.base_path).name))

    def test_retry_gpu_failure_keeps_first_candidate(self):
        with fake_factory(fail_render=2, visual=ContinuityQCResult(passed=False, note="Identity drift")) as state:
            result = factory.run_factory_generation(request(continuity_max_retries=1), workspace_id="audit")
            self.assertEqual(result.visual_qc_status, "failed")
            self.assertTrue(state.paths[0].exists())
            self.assertTrue(workspace_owns_generated_file("audit", state.paths[0].name))
            self.assertTrue(any("regeneration failed" in text for text in result.continuity_warnings))

    def test_strict_qc_runs_when_continuity_prompting_off(self):
        with fake_factory() as state:
            result = factory.run_factory_generation(request(continuity_mode="off"), workspace_id="audit")
            state.inspector.assert_called_once()
            self.assertEqual(result.visual_qc_status, "passed")

    def test_unchecked_video_cannot_publish(self):
        with fake_factory() as state:
            result = factory.run_factory_generation(request(continuity_qc_mode="off", publish_to_youtube=True), workspace_id="audit")
            state.publisher.assert_not_called()
            self.assertEqual(result.visual_qc_status, "not_checked")

    def test_passed_muted_video_can_publish(self):
        with fake_factory() as state:
            result = factory.run_factory_generation(request(publish_to_youtube=True), workspace_id="audit")
            state.publisher.assert_called_once()
            self.assertIsNotNone(result.youtube_url)

    def test_failed_audio_repair_preserves_video_and_reason(self):
        with fake_factory(audio=AudioQCResult(passed=False, note="Wrong dialogue"), fail_retake=True) as state:
            progress = Mock()
            result = factory.run_factory_generation(request(audio_mode="native"), workspace_id="audit", progress=progress)
            self.assertTrue(any('Correcting audio on the GPU' in call.args[2] for call in progress.call_args_list))
            self.assertEqual(result.audio_qc_status, "failed")
            self.assertTrue(state.paths[0].exists())
            self.assertTrue(any("repair unavailable" in text for text in result.audio_warnings))
            prompt = state.provider.retake_audio.call_args.kwargs["prompt"]
            self.assertIn("Welcome to Triven Cinema.", prompt)
            self.assertNotIn("VISUAL INTEGRITY", prompt)

    def test_mastering_failure_returns_saved_source(self):
        with fake_factory(fail_delivery=True) as state:
            result = factory.run_factory_generation(request(publish_to_youtube=True), workspace_id="audit")
            self.assertFalse(result.delivery_complete)
            self.assertTrue((state.root / result.final_filename).exists())
            self.assertTrue(result.warnings)
            state.publisher.assert_not_called()

    def test_upload_failure_returns_completed_video(self):
        with fake_factory(fail_publish=True) as state:
            result = factory.run_factory_generation(request(publish_to_youtube=True), workspace_id="audit")
            self.assertTrue((state.root / result.final_filename).exists())
            self.assertTrue(any("publication failed" in text for text in result.warnings))

    def test_failed_scene_does_not_become_next_identity_anchor(self):
        with fake_factory(count=2, visual=ContinuityQCResult(passed=False, note="Identity drift")) as state:
            result = factory.run_factory_generation(request(target_duration_seconds=30, continuity_max_retries=0), workspace_id="audit")
            self.assertEqual(len(result.scene_results), 2)
            self.assertIsNone(state.renderer.call_args_list[1].kwargs["reference_image_path"])

    def test_later_scene_failure_retains_checkpoint_and_owned_media(self):
        with fake_factory(count=2, fail_render=2) as state:
            payload = request(target_duration_seconds=30, continuity_mode="off")
            job_id = submit_job("factory", {"workspace_id": "audit"},
                                lambda _: factory.run_factory_generation(payload, workspace_id="audit").model_dump())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                job = get_job(job_id)
                if job["status"] == "failed":
                    break
                time.sleep(.01)
            self.assertEqual(job["status"], "failed")
            self.assertEqual(len(job["result"]["scene_results"]), 1)
            self.assertTrue(job["assets"])
            self.assertTrue(state.paths[0].exists())


class QCValidationTests(unittest.TestCase):
    def test_base_with_failed_or_unavailable_qc_never_replaces_refined_take(self):
        for verdict in (ContinuityQCResult(passed=False, note="Wrong face"),
                        ContinuityQCResult(passed=True, skipped=True, note="Provider unavailable")):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp:
                take = refined_take(Path(tmp))
                failed = ContinuityQCResult(passed=False, note="Texture artifacts")
                result, qc, attempt = review.recover_before_refinement(
                    take, failed, inspect=Mock(return_value=verdict), workspace_id=None, scene_index=0,
                )
                self.assertIs(result, take)
                self.assertIs(qc, failed)
                self.assertFalse(attempt["selected"])
                self.assertTrue(Path(take.base_path).exists())

    def test_base_is_not_rechecked_when_refined_qc_passes_or_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            take = refined_take(Path(tmp))
            inspect = Mock()
            for verdict in (ContinuityQCResult(passed=True), ContinuityQCResult(passed=False, skipped=True), None):
                result, qc, attempt = review.recover_before_refinement(
                    take, verdict, inspect=inspect, workspace_id=None, scene_index=0,
                )
                self.assertIs(result, take)
                self.assertIsNone(attempt)
            inspect.assert_not_called()

    def test_base_recovery_preserves_paid_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            take = refined_take(Path(tmp))
            result, _, _ = review.recover_before_refinement(
                take, ContinuityQCResult(passed=False), inspect=Mock(return_value=ContinuityQCResult(passed=True)),
                workspace_id=None, scene_index=0,
            )
            self.assertEqual(result.render_seconds, 120)
            self.assertEqual(result.wall_seconds, 125)
            self.assertFalse(result.detail_refined)
            self.assertIsNone(result.base_path)

    def test_incomplete_visual_verdict_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); frame = root / "frame.png"; frame.write_bytes(b"image")
            with patch.object(settings, "gemini_api_key", "dummy"), patch.object(continuity_qc, "extract_qc_frames", return_value=[frame]), patch.object(continuity_qc, "generate_content", return_value=SimpleNamespace(payload={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})):
                result = continuity_qc.evaluate_scene_cardinality(root / "scene.mp4", entity_locks=[], visible_entity_counts={}, character_bible="", scene_prompt="A studio.")
            self.assertTrue(result.skipped)

    def test_qc_frame_probe_failure_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "gemini_api_key", "dummy"), patch.object(continuity_qc, "extract_qc_frames", side_effect=RuntimeError("probe timeout")):
            result = continuity_qc.evaluate_scene_cardinality(Path(tmp) / "scene.mp4", entity_locks=[], visible_entity_counts={}, character_bible="", scene_prompt="A studio.")
            self.assertTrue(result.skipped)

    def test_incomplete_audio_verdict_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(settings, "gemini_api_key", "dummy"), patch.object(audio_qc, "_extract_audio", side_effect=lambda source, dest: dest.write_bytes(b"audio")), patch.object(audio_qc, "generate_content", return_value=SimpleNamespace(payload={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})):
                result = audio_qc.evaluate_scene_audio(root / "scene.mp4", scene_prompt="A studio.", spoken_script="Hello.")
            self.assertTrue(result.skipped)

    def test_missing_requested_audio_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(settings, "gemini_api_key", "dummy"), patch.object(audio_qc, "_extract_audio", side_effect=audio_qc.MissingAudioError("No audio stream")):
            result = audio_qc.evaluate_scene_audio(Path(tmp) / "scene.mp4", scene_prompt="A studio.", spoken_script="Hello.")
            self.assertFalse(result.passed)
            self.assertFalse(result.skipped)


class DirectAndCompositionTests(unittest.TestCase):
    def test_good_base_avoids_another_direct_gpu_render(self):
        with fake_factory() as state, ExitStack() as stack:
            take = refined_take(state.root)
            state.renderer.side_effect = None
            state.renderer.return_value = take
            state.inspector.side_effect = [ContinuityQCResult(passed=False, artifact_detected=True), ContinuityQCResult(passed=True)]
            for name, value in {
                "GENERATED_DIR": state.root, "ensure_minimum_free_disk": Mock(),
                "get_video_provider": Mock(return_value=state.provider), "render_long_clip": state.renderer,
                "evaluate_scene_cardinality": state.inspector,
                "prepare_delivery": Mock(side_effect=lambda path, **_: path),
                "_media_info": Mock(return_value=MediaInfo(duration_seconds=15, has_audio=True)),
                "estimate_gpu_cost": Mock(return_value=(0., 0., "test")), "record_generation_metric": Mock(),
            }.items():
                stack.enter_context(patch.object(generations, name, value))
            result = generations._generate_video_impl(VideoGenerationRequest(
                prompt="A presenter in a studio.", duration_seconds=15, provider="modal", audio_mode="mute", continuity_max_retries=1,
            ))
            state.renderer.assert_called_once()
            self.assertEqual(result.visual_qc_status, "passed")
            self.assertEqual(result.filename, Path(take.base_path).name)
            self.assertFalse(result.detail_refined)

    def test_direct_audio_qc_receives_immutable_script(self):
        with fake_factory() as state, ExitStack() as stack:
            for name, value in {
                "GENERATED_DIR": state.root, "ensure_minimum_free_disk": Mock(),
                "get_video_provider": Mock(return_value=state.provider), "render_long_clip": state.renderer,
                "evaluate_scene_cardinality": state.inspector,
                "prepare_delivery": Mock(side_effect=lambda path, **_: path),
                "_media_info": Mock(return_value=MediaInfo(duration_seconds=15, has_audio=True)),
                "estimate_gpu_cost": Mock(return_value=(0., 0., "test")), "record_generation_metric": Mock(),
            }.items():
                stack.enter_context(patch.object(generations, name, value))
            inspector = stack.enter_context(patch.object(review, "evaluate_scene_audio", return_value=AudioQCResult(passed=True, note="Exact script heard")))
            result = generations._generate_video_impl(VideoGenerationRequest(
                prompt="A presenter with consistent face identity and lighting.",
                spoken_script="Only these spoken words.", duration_seconds=15, provider="modal",
                audio_mode="native", continuity_mode="off", continuity_max_retries=0,
            ))
            self.assertEqual(result.audio_qc_status, "passed")
            self.assertEqual(inspector.call_args.kwargs["spoken_script"], "Only these spoken words.")
            self.assertEqual(result.audio_qc["note"], "Exact script heard")

    def test_direct_retry_failure_keeps_rejected_candidate(self):
        with fake_factory(fail_render=2, visual=ContinuityQCResult(passed=False, note="Drift")) as state, ExitStack() as stack:
            for name, value in {
                "GENERATED_DIR": state.root, "ensure_minimum_free_disk": Mock(),
                "get_video_provider": Mock(return_value=state.provider), "render_long_clip": state.renderer,
                "evaluate_scene_cardinality": state.inspector,
                "prepare_delivery": Mock(side_effect=lambda path, **_: path),
                "_media_info": Mock(return_value=MediaInfo(duration_seconds=15, has_audio=False)),
                "estimate_gpu_cost": Mock(return_value=(0., 0., "test")), "record_generation_metric": Mock(),
            }.items():
                stack.enter_context(patch.object(generations, name, value))
            result = generations._generate_video_impl(VideoGenerationRequest(
                prompt="A presenter in a studio.", duration_seconds=15, provider="modal", audio_mode="mute", continuity_max_retries=1,
            ))
            self.assertEqual(result.visual_qc_status, "failed")
            self.assertTrue(state.paths[0].exists())

    def test_composition_requires_every_source_to_be_owned(self):
        from app.schemas.generation import CombineScenesRequest
        with tempfile.TemporaryDirectory() as tmp, patch.object(generations, "ensure_minimum_free_disk"):
            root = Path(tmp).resolve(); path = root / "not-owned.mp4"; path.write_bytes(b"video")
            with patch.object(generations, "GENERATED_DIR", root), patch.object(generations, "combine_videos") as combine:
                with self.assertRaisesRegex(ValueError, "this account"):
                    generations._combine_impl(CombineScenesRequest(scene_video_urls=[path.name]), "other-account")
                combine.assert_not_called()

    def test_composition_inherits_persisted_qc(self):
        from app.schemas.generation import CombineScenesRequest
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp).resolve(); source = root / "owned-combine-source.mp4"; source.write_bytes(b"video")
            review.register_render("combine-audit", source, visual_qc_status="failed", audio_qc_status="passed")
            stack.enter_context(patch.object(generations, "GENERATED_DIR", root))
            stack.enter_context(patch.object(generations, "ensure_minimum_free_disk"))
            stack.enter_context(patch.object(generations, "combine_videos", side_effect=lambda sources, target: target.write_bytes(b"combined")))
            stack.enter_context(patch.object(generations, "prepare_delivery", side_effect=lambda source, **_: source))
            stack.enter_context(patch.object(generations, "_media_info", return_value=MediaInfo(duration_seconds=15, has_audio=True)))
            result = generations._combine_impl(CombineScenesRequest(scene_video_urls=[source.name]), "combine-audit")
            self.assertEqual(result.visual_qc_status, "failed")
            self.assertEqual(result.audio_qc_status, "passed")
            self.assertTrue(workspace_owns_generated_file("combine-audit", result.final_filename))


class AudioCorrectionEndpointTests(unittest.TestCase):
    def test_audio_correction_requires_ownership_before_submission(self):
        from app.schemas.generation import AudioRetakeRequest
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp).resolve(); source = root / "other-user-video.mp4"; source.write_bytes(b"video")
            stack.enter_context(patch.object(generations, "GENERATED_DIR", root))
            stack.enter_context(patch.object(generations, "ensure_workspace", return_value="repair-user"))
            submit = stack.enter_context(patch.object(generations, "submit_generation_request"))
            with self.assertRaises(HTTPException) as error:
                generations.create_audio_retake_job(AudioRetakeRequest(filename=source.name, spoken_script="Hello"), None, None)
            self.assertEqual(error.exception.status_code, 404)
            submit.assert_not_called()

    def test_audio_correction_preserves_visual_verdict_and_charges_duration(self):
        from app.schemas.generation import AudioRetakeRequest
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp).resolve(); source = root / f"{root.name}-owned.mp4"; source.write_bytes(b"original")
            repaired = root / f"{root.name}-repaired.mp4"
            review.register_render("repair-user", source, visual_qc_status="failed", visual_qc={"status": "failed", "note": "Identity drift"})
            def retake(**kwargs):
                repaired.write_bytes(b"original picture, repaired audio")
                return SimpleNamespace(path=str(repaired))
            provider = SimpleNamespace(name="modal", supports_audio_retake=True, retake_audio=Mock(side_effect=retake))
            captured = {}
            def submit(kind, payload, owner, runner, **kwargs):
                captured.update(kind=kind, owner=owner, **kwargs)
                return runner("test-job")
            for name, value in {
                "GENERATED_DIR": root, "ensure_workspace": Mock(return_value="repair-user"),
                "get_video_provider": Mock(return_value=provider), "update_job": Mock(),
                "_media_info": Mock(return_value=MediaInfo(duration_seconds=15, has_audio=True)),
                "submit_generation_request": Mock(side_effect=submit),
            }.items():
                stack.enter_context(patch.object(generations, name, value))
            stack.enter_context(patch.object(review, "evaluate_scene_audio", return_value=AudioQCResult(passed=True, note="Correct speech")))
            stack.enter_context(patch.object(settings, "factory_audio_qc_enabled", True))
            result = generations.create_audio_retake_job(AudioRetakeRequest(filename=source.name, spoken_script="Only this sentence."), None, None)
            self.assertEqual(result["visual_qc_status"], "failed")
            self.assertEqual(result["audio_qc_status"], "passed")
            self.assertEqual(captured["charge_seconds"], 15)
            self.assertTrue(source.exists())
            self.assertTrue(workspace_owns_generated_file("repair-user", repaired.name))
            self.assertIn("Only this sentence.", provider.retake_audio.call_args.kwargs["prompt"])
            provider.retake_audio.assert_called_once()


if __name__ == "__main__":
    unittest.main()
