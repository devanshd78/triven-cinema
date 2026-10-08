import json
import unittest
from unittest.mock import patch

from app.services.dialogue_service import build_audio_prompt, extract_spoken_script, split_spoken_script
from app.services.scene_planner import create_prompt_only_plan, create_scene_plan


class DialoguePlanningTests(unittest.TestCase):
    def test_hindi_sentences_split_without_dropping_or_repeating_words(self):
        script = "पहले स्क्रीन की बात करते हैं। अब कैमरे की तरफ आते हैं।"
        scenes = split_spoken_script(script, 2)
        self.assertEqual(scenes, ["पहले स्क्रीन की बात करते हैं।", "अब कैमरे की तरफ आते हैं।"])
        self.assertEqual(" ".join(scenes), script)

    def test_short_legacy_utterance_is_not_repeated_to_fill_scenes(self):
        plan = create_prompt_only_plan('@char1 says "Welcome to our studio."', 4, target_scene_duration_seconds=15)
        self.assertEqual([s.spoken_script for s in plan.scenes], ["Welcome to our studio.", "", "", ""])
        self.assertTrue(all("Welcome to our studio." not in s.prompt for s in plan.scenes))

    def test_closing_dialogue_survives_visual_compaction(self):
        prompt = " ".join(["The studio contains a wooden desk and a soft background light."] * 35)
        prompt += ' The presenter says "The password is marigold."'
        plan = create_prompt_only_plan(prompt, 1, target_scene_duration_seconds=15)
        self.assertEqual(plan.scenes[0].spoken_script, "The password is marigold.")

    def test_explicit_script_is_never_rewritten_by_gemini(self):
        payload = {"character_bible": "A stable adult presenter with black hair.",
                   "style_bible": "A photoreal studio scene with consistent lighting.",
                   "scenes": [{"title": "Greeting", "prompt": "A presenter at a desk. " * 50 + 'He says "Invented line."',
                               "duration_seconds": 15, "visible_entity_counts": {}}]}
        with patch("app.services.scene_planner.generate_content") as remote:
            remote.return_value.payload = {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}
            plan = create_scene_plan("A presenter at a desk.", 1, force_ai=True, spoken_script="Welcome home.")
        self.assertEqual(plan.scenes[0].spoken_script, "Welcome home.")
        self.assertNotIn("Invented line", plan.scenes[0].prompt)

    def test_explicit_script_overrides_legacy_speech(self):
        plan = create_prompt_only_plan('The presenter says "Old words."', 1, spoken_script="New words.")
        self.assertEqual(plan.scenes[0].spoken_script, "New words.")
        self.assertNotIn("Old words", plan.scenes[0].prompt)

    def test_script_word_sequence_preserved_across_many_scenes(self):
        script = " ".join(f"Sentence {i} stays exact." for i in range(35))
        scenes = split_spoken_script(script, 20)
        self.assertEqual(" ".join(scenes).split(), script.split())

    def test_unpunctuated_narration_preserves_all_words(self):
        script = " ".join(f"word{i}" for i in range(101))
        self.assertEqual(" ".join(split_spoken_script(script, 4)).split(), script.split())

    def test_visual_quotes_do_not_become_dialogue(self):
        self.assertEqual(extract_spoken_script('Preserve "face identity" and "reference image".'), "")

    def test_prompt_compiler_has_explicit_silent_state(self):
        result = build_audio_prompt("Same face identity and reference image.", "", "Quiet room tone.")
        self.assertNotIn("NEVER SPOKEN", result)
        self.assertNotIn("[SPOKEN SCRIPT]", result)
        self.assertIn("without speech", result)

    def test_unicode_legacy_dialogue_is_preserved(self):
        plan = create_prompt_only_plan('The presenter says “नमस्ते दोस्तों।”', 1)
        self.assertEqual(plan.scenes[0].spoken_script, "नमस्ते दोस्तों।")


if __name__ == "__main__":
    unittest.main()
