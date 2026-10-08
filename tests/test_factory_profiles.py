import unittest
from unittest.mock import patch

from app.core.config import settings
from app.schemas.factory import FactoryGenerationRequest
from app.schemas.generation import ScenePlanRequest, VideoGenerationRequest
from app.services.video_combiner import delivery_dimensions
from app.services.video_profiles import duration_profile, validate_factory_scene_duration, validate_scene_duration
from app.services.scene_planner import create_prompt_only_plan, create_scene_plan
from app.services.factory_service import (
    _factory_scene_durations,
    _qc_unavailable_is_fatal,
    _scene_seed,
    _wants_single_continuous_shot,
)


class FactoryProfileTests(unittest.TestCase):
    def test_delivery_4k_dimensions(self):
        self.assertEqual(delivery_dimensions("16:9", "4k"), (3840, 2160))
        self.assertEqual(delivery_dimensions("9:16", "4k"), (2160, 3840))
        self.assertEqual(delivery_dimensions("1:1", "4k"), (2160, 2160))

    def test_default_duration_profile(self):
        profile = duration_profile()
        self.assertGreaterEqual(profile["1080p"], 30)
        self.assertGreaterEqual(profile["4k"], 15)

    def test_schema_accepts_thirty_second_1080p_scene(self):
        request = VideoGenerationRequest(
            prompt="A cinematic fox walks through fresh snow at sunrise.",
            quality="1080p",
            duration_seconds=30,
        )
        validate_scene_duration(quality=request.quality, duration_seconds=request.duration_seconds)

    def test_factory_accepts_multi_minute_target(self):
        request = FactoryGenerationRequest(
            prompt="A cinematic expedition progresses across a snowy forest with synchronized natural sound.",
            target_duration_seconds=120,
            scene_duration_seconds=30,
            quality="1080p",
        )
        self.assertEqual(request.target_duration_seconds, 120)

    def test_factory_defaults_to_twenty_second_single_pass_scenes(self):
        request = FactoryGenerationRequest(
            prompt="A cinematic expedition progresses across a snowy forest with synchronized natural sound."
        )
        self.assertEqual(request.scene_duration_seconds, 20)
        self.assertEqual(request.continuity_qc_mode, "strict")
        self.assertEqual(request.continuity_strength, 0.95)
        self.assertEqual(request.realism_profile, "real_skin")


    def test_direct_render_keeps_backward_compatible_realism_default(self):
        request = VideoGenerationRequest(
            prompt="A photoreal presenter speaks to camera in a bright home studio."
        )
        self.assertEqual(request.realism_profile, "standard")

    def test_factory_rejects_sub_fifteen_second_scene(self):
        with self.assertRaises(ValueError):
            FactoryGenerationRequest(
                prompt="A cinematic expedition progresses across a snowy forest with synchronized natural sound.",
                scene_duration_seconds=10,
            )

    def test_factory_validates_standard_fifteen_and_twenty_second_scenes(self):
        validate_factory_scene_duration(quality="1080p", duration_seconds=15)
        validate_factory_scene_duration(quality="1080p", duration_seconds=20)

    def test_factory_allows_user_selected_extended_1080p_scene_lengths(self):
        validate_factory_scene_duration(quality="1080p", duration_seconds=21)
        validate_factory_scene_duration(quality="1080p", duration_seconds=25)
        validate_factory_scene_duration(quality="1080p", duration_seconds=30)

    def test_factory_keeps_4k_at_fifteen_seconds(self):
        validate_factory_scene_duration(quality="4k", duration_seconds=15)
        with self.assertRaises(ValueError):
            validate_factory_scene_duration(quality="4k", duration_seconds=20)

    def test_factory_does_not_create_sub_fifteen_second_tail_scene(self):
        self.assertEqual(_factory_scene_durations(30, 20, 15), [15.0, 15.0])
        self.assertEqual(_factory_scene_durations(60, 20, 15), [20.0, 20.0, 20.0])
        self.assertEqual(_factory_scene_durations(60, 30, 15), [30.0, 30.0])


    def test_continuous_shot_intent_is_detected(self):
        self.assertTrue(_wants_single_continuous_shot("Filmed as one continuous static shot with no cuts."))
        self.assertTrue(_wants_single_continuous_shot("A single take talking-head creator video."))
        self.assertFalse(_wants_single_continuous_shot("A montage of several cinematic scenes."))

    def test_qc_provider_outage_does_not_discard_render_by_default(self):
        self.assertTrue(settings.factory_qc_fail_open_on_unavailable)
        self.assertFalse(_qc_unavailable_is_fatal(True))
        self.assertFalse(_qc_unavailable_is_fatal(False))

    def test_qc_provider_outage_can_be_configured_fail_closed(self):
        with patch.object(settings, "factory_qc_fail_open_on_unavailable", False):
            self.assertFalse(_qc_unavailable_is_fatal(True))
            self.assertFalse(_qc_unavailable_is_fatal(False))

    def test_factory_uses_distinct_deterministic_seed_per_scene(self):
        seeds = [_scene_seed(42, index, 0) for index in range(6)]
        self.assertEqual(len(set(seeds)), 6)
        self.assertEqual(seeds, [_scene_seed(42, index, 0) for index in range(6)])
        self.assertNotEqual(_scene_seed(42, 2, 0), _scene_seed(42, 2, 1))

    def test_factory_and_storyboard_accept_long_manuscripts(self):
        manuscript = "Vrindavan story beat. " * 1300
        self.assertGreater(len(manuscript), 8000)
        self.assertLess(len(manuscript), 50000)
        factory = FactoryGenerationRequest(prompt=manuscript)
        storyboard = ScenePlanRequest(prompt=manuscript, scene_count=8)
        self.assertEqual(factory.prompt, manuscript)
        self.assertEqual(storyboard.prompt, manuscript)

    def test_individual_render_keeps_bounded_prompt_limit(self):
        with self.assertRaises(ValueError):
            VideoGenerationRequest(prompt="x" * 8001)


    def test_prompt_only_single_shot_preserves_wardrobe_dialogue_and_realism_constraints(self):
        prompt = (
            "Live-action premium cinema camera shot. The presenter has real skin with visible pores and long wavy dark-brown hair. "
            "She wears a chunky cable-knit sweater in wide horizontal bands of white and soft lavender with visible wool fibers. "
            "The camera is locked at eye level in a medium shot behind a pale oak table. "
            + "Natural studio production detail. " * 120
            + 'She says: "Hey everyone! The new Mac mini and Mac Studio are finally here." '
            + "A large soft key light from front left keeps natural skin tones. Audio is clean studio voice with no music."
        )
        plan = create_prompt_only_plan(prompt, 1, target_scene_duration_seconds=30)
        scene_prompt = plan.scenes[0].prompt.lower()
        self.assertIn("cable-knit sweater", scene_prompt)
        self.assertIn("white and soft lavender", scene_prompt)
        self.assertIn("hey everyone", plan.scenes[0].spoken_script.lower())
        self.assertNotIn("hey everyone", scene_prompt)
        self.assertIn("visible pores", scene_prompt)
        self.assertEqual(plan.scenes[0].duration_seconds, 30)

    def test_factory_storyboard_is_paced_for_long_scene(self):
        plan = create_scene_plan(
            "A cinematic expedition progresses across a snowy forest with synchronized natural sound.",
            1,
            "16:9",
            target_scene_duration_seconds=30,
        )
        self.assertEqual(plan.scenes[0].duration_seconds, 30)


if __name__ == "__main__":
    unittest.main()
