#!/usr/bin/env python3
"""Run regression tests against a temporary source copy with empty runtime state."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='triven-tests-') as temporary:
        target = Path(temporary) / 'repo'
        shutil.copytree(ROOT, target, ignore=shutil.ignore_patterns(
            '.git', '.venv', 'storage', 'node_modules', '.next', '.env', '.env.local',
            '__pycache__', '*.pyc', '*.orig', '*.rej', '*.zip', '.DS_Store',
        ))
        env = os.environ.copy()
        env.update(PYTHONPATH=f'{target}:{target / "services/api"}', APP_ENV='development',
                   AUTH_ENABLED='true', DEMO_AUTH_SHOW_OTP='true', BILLING_ENABLED='false',
                   BILLING_ENFORCE_CREDITS='false', YOUTUBE_ENABLED='false',
                   TRIVEN_SECRET_KEY='isolated-tests-secret-never-for-production',
                   GEMINI_API_KEY='', HF_TOKEN='', MODAL_TOKEN_ID='', MODAL_TOKEN_SECRET='')
        return subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
                              cwd=target, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
