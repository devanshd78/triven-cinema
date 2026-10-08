import unittest

from app.services.prompt_quality import evaluate_plan_prompt_coverage
from app.services.continuity_service import compose_render_integrity_prompt


class PromptQualityTests(unittest.TestCase):
    def test_render_guards_keep_the_requested_shot_without_inventing_visual_content(self):
        scene = "@char3 holds one matte-black smartphone over a wooden desk in a technology studio."
        compiled = compose_render_integrity_prompt(scene, realism_profile="real_skin", prompt_wardrobe_authoritative=True)
        self.assertTrue(compiled.startswith(scene))
        self.assertIn("WARDROBE AUTHORITY", compiled)
        self.assertLess(len(compiled.split()) - len(scene.split()), 100)
        for unwanted in ("microphone", "CGI gloss", "extra teeth", "floating objects", "porcelain skin"):
            self.assertNotIn(unwanted, compiled)
        self.assertEqual(compose_render_integrity_prompt(compiled), compiled)

    def test_reports_full_coverage_when_terms_are_preserved(self):
        report = evaluate_plan_prompt_coverage(
            "Astronaut walks across a red Martian desert with dust and mountains",
            [
                "An astronaut walks across a red Martian desert while fine dust moves near distant mountains."
            ],
        )
        self.assertGreaterEqual(report["coverage_score"], 0.9)
        self.assertEqual(report["missing_terms"], [])

    def test_reports_missing_terms(self):
        report = evaluate_plan_prompt_coverage(
            "Astronaut walks across a red Martian desert with dust and mountains",
            ["A person walks across an empty desert."],
        )
        self.assertLess(report["coverage_score"], 1.0)
        self.assertIn("astronaut", report["missing_terms"])


if __name__ == "__main__":
    unittest.main()
