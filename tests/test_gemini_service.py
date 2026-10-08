import unittest
from unittest.mock import Mock, patch
import httpx

from app.services.gemini_service import generate_content


class GeminiServiceTests(unittest.TestCase):
    def test_shared_budget_stops_retries_across_all_models(self):
        elapsed = [0.0]
        timeouts = []
        def timeout(*args, **kwargs):
            seconds = kwargs['timeout'].read
            timeouts.append(seconds)
            elapsed[0] += seconds
            raise httpx.ReadTimeout('provider stalled')
        with (
            patch('app.services.gemini_service.settings.gemini_api_key', 'test-key'),
            patch('app.services.gemini_service.settings.gemini_model', 'primary'),
            patch('app.services.gemini_service.settings.gemini_fallback_models', 'fallback-one,fallback-two'),
            patch('app.services.gemini_service.time.monotonic', side_effect=lambda: elapsed[0]),
            patch('app.services.gemini_service.httpx.post', side_effect=timeout) as post,
        ):
            with self.assertRaisesRegex(RuntimeError, 'retry time budget exhausted'):
                generate_content(parts=[{'text': 'inspect audio'}], timeout_seconds=30,
                                 total_timeout_seconds=45, max_attempts_per_model=1)
        self.assertEqual(timeouts, [30, 15])
        self.assertEqual(post.call_count, 2)

    def test_quality_check_falls_back_without_repeating_failed_primary(self):
        unavailable = Mock(status_code=503)
        unavailable.json.return_value = {'error': {'status': 'UNAVAILABLE'}}
        success = Mock(status_code=200)
        success.json.return_value = {'candidates': []}
        with (
            patch('app.services.gemini_service.settings.gemini_api_key', 'test-key'),
            patch('app.services.gemini_service.settings.gemini_model', 'primary'),
            patch('app.services.gemini_service.settings.gemini_fallback_models', 'fallback'),
            patch('app.services.gemini_service.settings.gemini_max_attempts_per_model', 3),
            patch('app.services.gemini_service.httpx.post', side_effect=[unavailable, success]),
            patch('app.services.gemini_service.time.sleep') as sleep,
        ):
            result = generate_content(parts=[{'text': 'inspect video'}], total_timeout_seconds=45, max_attempts_per_model=1)
        self.assertEqual(result.model, 'fallback')
        self.assertEqual(result.attempts, 2)
        sleep.assert_not_called()

    def test_backoff_does_not_extend_exhausted_budget(self):
        unavailable = Mock(status_code=503)
        unavailable.json.return_value = {'error': {'status': 'UNAVAILABLE'}}
        with (
            patch('app.services.gemini_service.settings.gemini_api_key', 'test-key'),
            patch('app.services.gemini_service.settings.gemini_retry_backoff_seconds', 5),
            patch('app.services.gemini_service.settings.gemini_max_attempts_per_model', 3),
            patch('app.services.gemini_service.time.monotonic', return_value=0),
            patch('app.services.gemini_service.httpx.post', return_value=unavailable) as post,
            patch('app.services.gemini_service.time.sleep') as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, 'retry time budget exhausted'):
                generate_content(parts=[{'text': 'plan'}], total_timeout_seconds=1)
        post.assert_called_once()
        sleep.assert_not_called()

    @patch("app.services.gemini_service.time.sleep")
    @patch("app.services.gemini_service.httpx.post")
    def test_retries_503_then_succeeds(self, post, sleep):
        unavailable = Mock(status_code=503)
        unavailable.json.return_value = {"error": {"status": "UNAVAILABLE"}}
        success = Mock(status_code=200)
        success.json.return_value = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
        post.side_effect = [unavailable, success]

        with (
            patch("app.services.gemini_service.settings.gemini_api_key", "test-key"),
            patch("app.services.gemini_service.settings.gemini_model", "gemini-primary"),
            patch("app.services.gemini_service.settings.gemini_fallback_models", "gemini-fallback"),
            patch("app.services.gemini_service.settings.gemini_max_attempts_per_model", 3),
            patch("app.services.gemini_service.settings.gemini_retry_backoff_seconds", 0.01),
        ):
            result = generate_content(parts=[{"text": "plan"}])

        self.assertEqual(result.model, "gemini-primary")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    @patch("app.services.gemini_service.time.sleep")
    @patch("app.services.gemini_service.httpx.post")
    def test_fails_over_after_primary_retries(self, post, _sleep):
        unavailable = Mock(status_code=503)
        unavailable.json.return_value = {"error": {"status": "UNAVAILABLE"}}
        success = Mock(status_code=200)
        success.json.return_value = {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}
        post.side_effect = [unavailable, unavailable, success]

        with (
            patch("app.services.gemini_service.settings.gemini_api_key", "test-key"),
            patch("app.services.gemini_service.settings.gemini_model", "gemini-primary"),
            patch("app.services.gemini_service.settings.gemini_fallback_models", "gemini-fallback"),
            patch("app.services.gemini_service.settings.gemini_max_attempts_per_model", 2),
            patch("app.services.gemini_service.settings.gemini_retry_backoff_seconds", 0.0),
        ):
            result = generate_content(parts=[{"text": "plan"}])

        self.assertEqual(result.model, "gemini-fallback")
        self.assertEqual(result.attempts, 3)
        self.assertIn("gemini-fallback", post.call_args_list[-1].args[0])


if __name__ == "__main__":
    unittest.main()
