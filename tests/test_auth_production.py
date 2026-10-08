import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core.config import settings
from app.services import auth_service as auth
from app.services.identity_service import sign_workspace_id, workspace_id_from_request
from app.api.routes.auth import request_login_otp, router as auth_router
from app.schemas.auth import RequestOtpRequest


class ProductionAuthTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for obj, name, value in [
            (auth, 'AUTH_DIR', root), (auth, 'AUTH_DB', root/'auth.sqlite3'), (auth, '_INITIALIZED', False),
            (settings, 'app_env', 'production'), (settings, 'auth_enabled', True),
            (settings, 'demo_auth_show_otp', False), (settings, 'triven_secret_key', 'test-secret-with-at-least-32-characters'),
            (settings, 'smtp_host', 'smtp.example.test'), (settings, 'smtp_from_email', 'login@example.test'),
            (settings, 'smtp_username', 'test-user'), (settings, 'smtp_password', 'test-password'),
            (settings, 'smtp_use_tls', True), (settings, 'smtp_use_ssl', False),
        ]:
            self.stack.enter_context(patch.object(obj, name, value))

    def test_production_demo_login_without_smtp_preserves_signed_account_session(self):
        app = FastAPI()
        app.include_router(auth_router, prefix='/auth')
        with patch.object(settings, 'demo_auth_show_otp', True), \
             patch.object(settings, 'smtp_host', ''), \
             patch.object(settings, 'smtp_from_email', ''), \
             patch.object(settings, 'smtp_use_tls', False), \
             patch.object(auth.smtplib, 'SMTP') as smtp, \
             patch.object(auth.smtplib, 'SMTP_SSL') as smtp_ssl, \
             TestClient(app, base_url='https://cinema.example.test') as client:
            auth.validate_auth_configuration()
            self.assertTrue(auth.demo_login_enabled())
            response = client.post('/auth/otp/request', json={'email': 'creator@example.test'})
            self.assertEqual(response.status_code, 200, response.text)
            challenge = response.json()
            self.assertTrue(challenge['demo_mode'])
            otp = challenge['demo_otp']
            self.assertEqual(len(otp), 6)
            self.assertTrue(otp.isdigit())
            wrong = '000000' if otp != '000000' else '999999'
            self.assertEqual(client.post('/auth/otp/verify', json={'email': 'creator@example.test', 'otp': wrong}).status_code, 400)
            response = client.post('/auth/otp/verify', json={'email': 'creator@example.test', 'otp': otp})
            self.assertEqual(response.status_code, 200, response.text)
            user = response.json()['user']
            cookies = response.headers.get_list('set-cookie')
            self.assertTrue(cookies)
            self.assertTrue(all('Secure' in cookie and 'HttpOnly' in cookie for cookie in cookies))
            self.assertEqual(client.get('/auth/me').json()['user']['id'], user['id'])
            self.assertEqual(client.post('/auth/otp/verify', json={'email': 'creator@example.test', 'otp': otp}).status_code, 400)
            client.post('/auth/logout')
            self.assertFalse(client.get('/auth/me').json()['authenticated'])
            next_otp = client.post('/auth/otp/request', json={'email': 'creator@example.test'}).json()['demo_otp']
            returning = client.post('/auth/otp/verify', json={'email': 'creator@example.test', 'otp': next_otp}).json()['user']
            self.assertEqual(returning, user)
            smtp.assert_not_called()
            smtp_ssl.assert_not_called()

    def test_production_demo_still_requires_auth_and_signing_key(self):
        with patch.object(settings, 'demo_auth_show_otp', True):
            with patch.object(settings, 'triven_secret_key', 'short'):
                with self.assertRaisesRegex(auth.AuthError, 'TRIVEN_SECRET_KEY'):
                    auth.validate_auth_configuration()
            with patch.object(settings, 'auth_enabled', False):
                with self.assertRaisesRegex(auth.AuthError, 'AUTH_ENABLED'):
                    auth.validate_auth_configuration()

    def test_smtp_configuration_is_required_only_when_demo_is_disabled(self):
        with patch.object(settings, 'smtp_host', ''):
            with self.assertRaises(auth.AuthDeliveryError):
                auth.validate_auth_configuration()
        with patch.object(settings, 'smtp_use_tls', False):
            with self.assertRaises(auth.AuthDeliveryError):
                auth.validate_auth_configuration()

    def test_production_delivers_email_without_returning_code(self):
        request = Request({'type': 'http', 'headers': [], 'client': ('127.0.0.1', 1234)})
        with patch.object(auth.smtplib, 'SMTP') as smtp:
            result = request_login_otp(RequestOtpRequest(email='creator@example.test'), request)
            connection = smtp.return_value.__enter__.return_value
            connection.starttls.assert_called_once()
            connection.login.assert_called_once_with('test-user', 'test-password')
            message = connection.send_message.call_args.args[0]
            self.assertEqual(message['To'], 'creator@example.test')
            self.assertIsNone(result.demo_otp)
            self.assertFalse(result.demo_mode)

    def test_delivery_failure_invalidates_challenge(self):
        with patch.object(auth.smtplib, 'SMTP', side_effect=OSError('connection unavailable')):
            with self.assertRaises(auth.AuthDeliveryError):
                auth.request_otp('creator@example.test')
        with auth._connect() as conn:
            self.assertEqual(conn.execute('SELECT count(*) FROM otp_challenges').fetchone()[0], 0)

    def test_resend_and_ip_limits(self):
        with patch.object(auth, '_deliver_otp'):
            auth.request_otp('creator@example.test')
            with self.assertRaises(auth.AuthRateLimitError):
                auth.request_otp('creator@example.test')
        with patch.object(settings, 'auth_otp_requests_per_hour', 1):
            auth.limit_otp_requests('client-one')
            with self.assertRaises(auth.AuthRateLimitError):
                auth.limit_otp_requests('client-one')
            auth.limit_otp_requests('client-two')

    def test_account_identity_overrides_stale_workspace_cookie(self):
        with patch.object(auth, '_deliver_otp'):
            _, otp, _ = auth.request_otp('creator@example.test')
        user = auth.verify_otp('creator@example.test', otp)
        token = auth.sign_auth_user(user)
        stale = sign_workspace_id('b'*32)
        request = Request({'type':'http','headers':[(b'cookie', f'triven_auth={token}; triven_workspace={stale}'.encode())]})
        self.assertEqual(workspace_id_from_request(request), user['workspace_id'])
        unsigned_in = Request({'type':'http','headers':[(b'cookie', f'triven_workspace={stale}'.encode())]})
        self.assertIsNone(workspace_id_from_request(unsigned_in))


if __name__ == '__main__':
    unittest.main()
