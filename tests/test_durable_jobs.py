import json
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.services import job_service as jobs


class DurableJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.patches = [
            patch.object(jobs, 'JOBS_DIR', root),
            patch.object(jobs, 'JOBS_DB', root / 'jobs.sqlite3'),
            patch.object(jobs, '_INITIALIZED', False),
            patch.object(jobs, '_EXECUTOR', self.executor),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        self.executor.shutdown(wait=True)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def wait(self, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = jobs.get_job(job_id)
            if job['status'] in ('completed', 'failed'):
                return job
            time.sleep(.01)
        self.fail('job did not finish')

    def test_assets_remain_owned_after_500_other_jobs_and_job_pruning(self):
        job_id = jobs.create_job('video', {'workspace_id': 'alice'})
        jobs.update_job(job_id, status='completed', result={'filename': 'retained.mp4'})
        with sqlite3.connect(jobs.JOBS_DB) as conn:
            for i in range(501):
                conn.execute('''INSERT INTO generation_jobs(id,job_type,status,stage,progress,message,payload_json,result_json,created_at,updated_at,workspace_id)
                    VALUES(?, 'video','completed','completed',100,'done','{}','{}','2099','2099','bob')''', (f'new-{i}',))
            old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
            conn.execute('UPDATE generation_jobs SET updated_at=? WHERE id=?', (old, job_id))
        self.assertTrue(jobs.workspace_owns_generated_file('alice', 'retained.mp4'))
        jobs.prune_old_jobs()
        self.assertIsNone(jobs.get_job(job_id))
        self.assertTrue(jobs.workspace_owns_generated_file('alice', 'retained.mp4'))
        self.assertFalse(jobs.workspace_owns_generated_file('bob', 'retained.mp4'))

    def test_exception_retains_checkpoint_and_asset_access(self):
        def runner(job_id):
            jobs.register_generated_asset('alice', 'base.mp4', metadata={'visual_qc_status': 'unavailable'})
            jobs.checkpoint_job_result({'scene_results': [{'filename': 'base.mp4'}]})
            raise RuntimeError('simulated delivery timeout')
        job_id = jobs.submit_job('video', {'workspace_id': 'alice'}, runner)
        result = self.wait(job_id)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['result']['scene_results'][0]['filename'], 'base.mp4')
        self.assertEqual(result['assets'][0]['filename'], 'base.mp4')
        self.assertTrue(jobs.workspace_owns_generated_file('alice', 'base.mp4'))
        self.assertIn('timed out', result['error'])

    def test_idempotent_submission_runs_once_and_is_scoped(self):
        calls = []
        payload = {'workspace_id': 'alice', 'request_id': 'retry-1', 'chat_id': 'chat-a'}
        one = jobs.submit_job('video', payload, lambda job_id: calls.append(job_id) or {'ok': True})
        two = jobs.submit_job('video', payload, lambda job_id: calls.append(job_id) or {'ok': True})
        self.assertEqual(one, two)
        self.wait(one)
        self.assertEqual(calls, [one])
        self.assertEqual(len(jobs.list_jobs('alice', chat_id='chat-a')), 1)
        self.assertEqual(jobs.list_jobs('bob'), [])
        self.assertEqual(jobs.list_jobs('alice', chat_id='chat-b'), [])
        with self.assertRaises(ValueError):
            jobs.submit_job('video', {**payload, 'prompt': 'different'}, lambda _: {})

    def test_reference_inputs_do_not_grant_ownership(self):
        job_id = jobs.create_job('video', {'workspace_id': 'alice'})
        jobs.update_job(job_id, result={'filename': 'own.mp4', 'reference_frame_filename': 'foreign.png', 'prompt': 'foreign.mp4'})
        self.assertTrue(jobs.workspace_owns_generated_file('alice', 'own.mp4'))
        self.assertFalse(jobs.workspace_owns_generated_file('alice', 'foreign.png'))
        self.assertFalse(jobs.workspace_owns_generated_file('alice', 'foreign.mp4'))
        with self.assertRaises(ValueError):
            jobs.register_generated_asset('bob', 'own.mp4')

    def test_migrates_legacy_ownership_once_without_latest_job_limit(self):
        with sqlite3.connect(jobs.JOBS_DB) as conn:
            conn.execute('''CREATE TABLE generation_jobs(id TEXT PRIMARY KEY,job_type TEXT,status TEXT,stage TEXT,progress INTEGER,message TEXT,payload_json TEXT,result_json TEXT,error TEXT,created_at TEXT,updated_at TEXT)''')
            conn.execute('INSERT INTO generation_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)', (
                'legacy','video','completed','completed',100,'done',json.dumps({'workspace_id':'alice'}),
                json.dumps({'final_filename':'legacy.mp4'}),None,'2020','2020'))
        jobs.initialize_job_store()
        self.assertTrue(jobs.workspace_owns_generated_file('alice', 'legacy.mp4'))
        self.assertFalse(jobs.workspace_owns_generated_file('bob', 'legacy.mp4'))

    def test_asset_qc_metadata_survives_job_pruning(self):
        jobs.register_generated_asset('alice','qc.mp4',metadata={'visual_qc_status':'failed'})
        jobs.register_generated_asset('alice','qc.mp4',metadata={'audio_qc_status':'passed'})
        self.assertEqual(jobs.generated_asset_metadata('alice','qc.mp4'), {'visual_qc_status':'failed','audio_qc_status':'passed'})
        with self.assertRaises(ValueError):
            jobs.generated_asset_metadata('bob','qc.mp4')


if __name__ == '__main__':
    unittest.main()
