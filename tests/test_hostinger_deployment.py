import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
from scripts import production_preflight


ROOT = Path(__file__).resolve().parents[1]


class HostingerDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.compose = yaml.safe_load((ROOT / "docker-compose.production.yml").read_text())
        self.services = self.compose["services"]

    def test_services_bind_only_to_loopback(self):
        self.assertNotIn("caddy", self.services)
        self.assertIn("127.0.0.1:3334:8000", self.services["api"]["ports"])
        self.assertIn("127.0.0.1:3333:3000", self.services["web"]["ports"])
        self.assertNotIn("ports", self.services["maintenance"])

    def test_host_nginx_routes_public_traffic_to_loopback_services(self):
        nginx = (ROOT / "deploy/hostinger/nginx.triven-cinema.conf").read_text()
        self.assertIn("server_name devansh.info;", nginx)
        self.assertIn("proxy_pass http://127.0.0.1:3333;", nginx)
        self.assertIn("proxy_pass http://127.0.0.1:3334;", nginx)
        self.assertNotIn("proxy_pass http://api:8000;", nginx)

    def test_api_and_maintenance_share_persistent_storage(self):
        self.assertIn("./storage:/app/storage", self.services["api"]["volumes"])
        self.assertIn("./storage:/app/storage", self.services["maintenance"]["volumes"])

    def test_production_template_protects_paid_render_path(self):
        text = (ROOT / ".env.production.example").read_text()
        self.assertIn('APP_ENV="production"', text)
        self.assertIn('VIDEO_PROVIDER="modal"', text)
        self.assertIn("ENABLE_SYNC_RENDER_ENDPOINTS=false", text)
        self.assertIn("JOB_WORKERS=1", text)
        self.assertIn("JOB_MAX_PENDING=3", text)
        self.assertIn("DEMO_AUTH_SHOW_OTP=true", text)
        self.assertIn("MAX_1080P_SCENE_SECONDS=30", text)
        self.assertIn("MAX_4K_SCENE_SECONDS=15", text)

    def test_demo_domain_is_pinned_to_devansh_info(self):
        env_text = (ROOT / ".env.production.example").read_text()
        nginx_text = (ROOT / "deploy/hostinger/nginx.triven-cinema.conf").read_text()
        self.assertIn('TRIVEN_DOMAIN="devansh.info"', env_text)
        self.assertIn('FRONTEND_URL="https://devansh.info"', env_text)
        self.assertIn("server_name devansh.info;", nginx_text)


class ProductionLoginPreflightTests(unittest.TestCase):
    def check_configuration(self, **overrides):
        env = {'AUTH_ENABLED': 'true', 'TRIVEN_SECRET_KEY': 'test-secret-with-at-least-32-characters'}
        env.update(overrides)
        with patch.object(production_preflight, 'ERRORS', []), redirect_stdout(io.StringIO()):
            production_preflight.check_auth_configuration(env)
            return list(production_preflight.ERRORS)

    def test_demo_production_needs_no_smtp(self):
        for flag in ('true', '1', 'yes'):
            with self.subTest(flag=flag):
                self.assertEqual(self.check_configuration(DEMO_AUTH_SHOW_OTP=flag, SMTP_USE_TLS='false'), [])
        self.assertEqual(self.check_configuration(), [])

    def test_email_login_still_checks_delivery_configuration(self):
        errors = self.check_configuration(DEMO_AUTH_SHOW_OTP='false', SMTP_USE_TLS='false')
        self.assertEqual(len(errors), 3)
        self.assertTrue(any('SMTP_HOST' in error for error in errors))
        self.assertEqual(self.check_configuration(DEMO_AUTH_SHOW_OTP='false', SMTP_HOST='smtp.example.test', SMTP_FROM_EMAIL='login@example.test'), [])

    def test_demo_login_keeps_auth_and_signing_key_checks(self):
        self.assertTrue(self.check_configuration(DEMO_AUTH_SHOW_OTP='true', TRIVEN_SECRET_KEY='short'))
        self.assertTrue(self.check_configuration(DEMO_AUTH_SHOW_OTP='true', AUTH_ENABLED='false'))


if __name__ == "__main__":
    unittest.main()
