import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from app.core.config import settings

from app.schemas.generation import EntityLock
from app.services.continuity_qc import evaluate_scene_cardinality


class ContinuityQCTests(unittest.TestCase):
    def test_off_mode_skips_without_touching_media(self):
        result = evaluate_scene_cardinality(
            Path("/does/not/need/to/exist.mp4"),
            entity_locks=[EntityLock(label="LEO", expected_count=1)],
            visible_entity_counts={"LEO": 1},
            character_bible="LEO is the recurring adventurer.",
            scene_prompt="LEO walks forward.",
            qc_mode="off",
        )
        self.assertTrue(result.skipped)
        self.assertFalse(result.passed)
        self.assertTrue(result.not_checked)

    def test_reference_images_are_labeled_as_inputs_not_generated_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.jpg"
            frame1 = root / "frame1.png"
            frame2 = root / "frame2.png"
            for filename in (reference, frame1, frame2):
                Image.new("RGB", (32, 32), (80, 90, 100)).save(filename)
            captured = {}

            def fake_gemini(*, parts, **_kwargs):
                captured["parts"] = parts
                return SimpleNamespace(payload={"candidates": [{"content": {"parts": [{"text": json.dumps({
                    "passed": True, "duplicate_detected": False, "identity_drift_detected": False,
                    "artifact_detected": False, "wardrobe_mismatch_detected": False,
                    "confidence": 0.95, "note": "No visible defects.", "violations": [],
                    "identity_violations": [], "artifact_violations": [], "wardrobe_violations": [],
                })}]}}]})

            with patch.object(settings, "gemini_api_key", "dummy-key"), patch.object(settings, "continuity_vision_qc_enabled", True), patch(
                "app.services.continuity_qc.extract_qc_frames", return_value=[frame1, frame2]
            ), patch("app.services.continuity_qc.generate_content", side_effect=fake_gemini):
                result = evaluate_scene_cardinality(
                    root / "scene.mp4",
                    entity_locks=[], visible_entity_counts={"PERSON": 1},
                    character_bible="One person.", scene_prompt="One presenter talking.",
                    qc_mode="strict", canonical_reference_paths=[("@char1", reference)],
                )
            self.assertTrue(result.passed)
            text_parts = [part["text"] for part in captured["parts"] if "text" in part]
            self.assertIn("CANONICAL ELEMENT REFERENCES:", text_parts)
            self.assertTrue(any("NOT frames of the generated video" in part for part in text_parts))
            self.assertEqual(sum("GENERATED CLIP SAMPLE 1/2" in part for part in text_parts), 1)
            self.assertEqual(sum("GENERATED CLIP SAMPLE 2/2" in part for part in text_parts), 1)


if __name__ == "__main__":
    unittest.main()
