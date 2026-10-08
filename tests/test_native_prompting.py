import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.dialogue_service import build_audio_prompt
from inference.prompting import PROMPT_FORMAT_VERSION, compile_native_prompt, normalize_ltx_prompt
from inference.providers.modal_ltx import ModalLTXProvider


def legacy_prompt(visual, script, sound="Quiet room tone."):
    return ("[VISUAL INSTRUCTIONS — NEVER SPOKEN]\n" + visual
            + "\n\n[SOUND DIRECTION — NEVER SPOKEN]\n" + sound
            + "\n\n[SPOKEN SCRIPT]\nSpeak exactly the following words once, in order; no other narration, "
            "instructions, repetition or invented speech. Synchronize the visible speaker's lips:\n"
            + json.dumps(script, ensure_ascii=False))


class NativePromptTests(unittest.TestCase):
    def test_scene_and_hindi_script_survive_without_application_headings(self):
        visual = "Reference sheet: Single character portrait.\n\nGenerated video: A presenter holds a phone in a studio.\n\n[TRIVEN VISUAL INTEGRITY]\nKeep the same face."
        script = "नमस्कार दोस्तों! आज हम इस स्मार्टफोन की स्क्रीन और कैमरे को करीब से देखेंगे।"
        result = build_audio_prompt(visual, script, "One clear Hindi male voice. No music.")
        self.assertTrue(result.startswith("Reference sheet:"))
        self.assertIn("Generated video: A presenter holds a phone in a studio.", result)
        self.assertIn("Keep the same face.", result)
        self.assertIn("One clear Hindi male voice. No music.", result)
        self.assertEqual(result.count(script), 1)
        self.assertIn('The speaker says "' + script + '".', result)
        for marker in ("NEVER SPOKEN", "[SPOKEN SCRIPT]", "[TRIVEN", "Speak exactly the following"):
            self.assertNotIn(marker, result)

    def test_worker_can_clean_legacy_api_requests_without_changing_speech(self):
        visual = "Reference sheet: One portrait.\n\nGenerated video: A person presents a smartphone."
        script = 'Okay… this is it.\nThe iPhone 18. I\'ve been waiting for this one.'
        old = legacy_prompt(visual, script)
        result = normalize_ltx_prompt("\n  " + old)
        self.assertEqual(result, compile_native_prompt(visual, script, "Quiet room tone."))
        self.assertIn(json.dumps(script, ensure_ascii=False), result)
        self.assertEqual(normalize_ltx_prompt(result), result)

    def test_legacy_audio_only_retake_preserves_dialogue(self):
        result = normalize_ltx_prompt('[SOUND DIRECTION — NEVER SPOKEN]\nWarm voice.\n\n[SPOKEN SCRIPT]\nSpeak exactly:\n"Hello."')
        self.assertEqual(result, 'Warm voice.\n\nThe speaker says "Hello.".')

    def test_legacy_silent_state_remains_silent(self):
        result = normalize_ltx_prompt("[VISUAL INSTRUCTIONS — NEVER SPOKEN]\nLeaves rustle.\n\n[SPOKEN SCRIPT]\nNo dialogue, narration, singing, chanting or speech-like vocal sounds. Use only requested ambience, Foley or music.")
        self.assertIn("Leaves rustle.", result)
        self.assertIn("without speech", result)
        self.assertNotIn("SPOKEN", result)

    def test_malformed_legacy_dialogue_fails_instead_of_silently_dropping_words(self):
        for payload in ('not valid quoted speech', '123', '{"script": "Hello"}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                normalize_ltx_prompt("[SPOKEN SCRIPT]\nSpeak exactly:\n" + payload)

    def test_requested_on_screen_text_is_not_stripped(self):
        visual = 'A poster reads "NEVER SPOKEN". A neon sign shows "[TRIVEN VISUAL INTEGRITY]".'
        self.assertEqual(normalize_ltx_prompt(visual), visual)

    def test_inline_quoted_dialogue_is_not_accidentally_silenced(self):
        result = build_audio_prompt('A presenter says "Welcome home."', "")
        self.assertIn('The speaker says "Welcome home.".', result)
        self.assertNotIn("without speech", result)

    def test_api_reports_prompt_format_to_verify_deployment(self):
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prompt_format"], PROMPT_FORMAT_VERSION)

    def test_bad_legacy_input_fails_before_modal_preflight_or_gpu(self):
        with patch.object(ModalLTXProvider, "preflight") as preflight, patch("inference.providers.modal_ltx.modal.Function.from_name") as lookup:
            with self.assertRaises(ValueError):
                ModalLTXProvider().generate("[SPOKEN SCRIPT]\ninvalid", 512, 512, 5, 42, "conv")
        preflight.assert_not_called()
        lookup.assert_not_called()
