import unittest

from app.services.job_service import safe_job_error


class InferenceErrorMessageTests(unittest.TestCase):
    def test_disabled_workspace_is_not_reported_as_missing_weights(self):
        message = safe_job_error(RuntimeError(
            'Modal inference preflight failed: workspace ac-private-workspace is disabled token=private-value'
        ))
        self.assertIn('Modal workspace is disabled', message)
        self.assertNotIn('ac-private-workspace', message)
        self.assertNotIn('private-value', message)
        self.assertNotIn('model weights', message)
        self.assertNotIn('modal deploy', message)

    def test_missing_readiness_function_explains_worker_deployment(self):
        message = safe_job_error(RuntimeError(
            "Modal inference preflight failed before GPU allocation. Lookup failed for Function 'preflight': "
            "Function 'preflight' not found on App 'triven-cinema-ltx'. token=private-value"
        ))
        self.assertIn('missing its readiness-check function', message)
        self.assertIn('modal deploy modal/app.py', message)
        self.assertIn('scripts/check_inference.py --all', message)
        self.assertNotIn('private-value', message)
        self.assertNotIn('model weights', message)

    def test_incompatible_worker_explains_matching_runtime(self):
        for detail in ('Modal worker protocol is incompatible', 'Modal worker LTX revision mismatch'):
            with self.subTest(detail=detail):
                message = safe_job_error(RuntimeError(detail))
                self.assertIn('does not match this application', message)
                self.assertIn('TRIVEN_LTX_REPO_REF', message)

    def test_weight_failure_does_not_claim_the_readiness_function_is_missing(self):
        message = safe_job_error(RuntimeError('Modal inference is not ready: Missing or empty model files: test.safetensors'))
        self.assertIn('model weights', message)
        self.assertNotIn('missing its readiness-check function', message)


if __name__ == '__main__':
    unittest.main()
