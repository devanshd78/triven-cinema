import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProductionRolloutTests(unittest.TestCase):
    def run_rollout(self, *args, fail_worker=False, fail_preflight=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            target = scripts / "deploy_hostinger.sh"
            shutil.copyfile(ROOT / "scripts/deploy_hostinger.sh", target)
            (root / ".env").write_text("TEST_CONFIG=true\n")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            log = root / "commands.txt"
            for name in ("python3", "docker", "curl"):
                executable = bin_dir / name
                executable.write_text(
                    '#!/bin/sh\nprintf "%s\\n" "$0 $*" >> "$ROLLOUT_TEST_LOG"\n'
                    'case "$*" in\n'
                    ' *"modal deploy"*) [ "$ROLLOUT_FAIL_WORKER" = 1 ] && exit 17 ;;\n'
                    ' *"check_inference.py"*) [ "$ROLLOUT_FAIL_PREFLIGHT" = 1 ] && exit 18 ;;\n'
                    'esac\nexit 0\n'
                )
                executable.chmod(0o755)
            env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                   "ROLLOUT_TEST_LOG": str(log), "ROLLOUT_FAIL_WORKER": str(int(fail_worker)),
                   "ROLLOUT_FAIL_PREFLIGHT": str(int(fail_preflight))}
            result = subprocess.run(["bash", str(target), *args], cwd=root, env=env,
                                    text=True, capture_output=True, timeout=10)
            return result, log.read_text() if log.exists() else ""

    def test_worker_deploy_and_check_use_same_compose_api_environment(self):
        result, log = self.run_rollout("--deploy-worker")
        self.assertEqual(result.returncode, 0, result.stderr)
        worker = "compose -f docker-compose.production.yml run --rm --no-deps api python -m modal deploy modal/app.py --strategy rolling"
        check = "compose -f docker-compose.production.yml run --rm --no-deps api python /app/scripts/check_inference.py --all --require-current-worker"
        self.assertIn(worker, log)
        self.assertIn(check, log)
        self.assertLess(log.index("backup_state.py"), log.index("build --pull"))
        self.assertLess(log.index(worker), log.index(check))
        self.assertLess(log.index(check), log.index("up -d"))

    def test_failed_worker_deployment_does_not_replace_running_api(self):
        result, log = self.run_rollout("--deploy-worker", fail_worker=True)
        self.assertEqual(result.returncode, 17)
        self.assertNotIn("up -d", log)

    def test_wrong_worker_preflight_stops_application_rollout(self):
        result, log = self.run_rollout("--deploy-worker", fail_preflight=True)
        self.assertEqual(result.returncode, 18)
        self.assertNotIn("up -d", log)

    def test_regular_rollout_does_not_silently_deploy_a_worker(self):
        result, log = self.run_rollout()
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("modal deploy", log)
        self.assertIn("--require-current-worker", log)
