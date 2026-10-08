"""Credit reservation + job persistence boundaries, entirely on temporary stores."""
import sqlite3
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

from app.core.config import settings
from app.schemas.generation import VideoGenerationRequest
from app.services import billing_service as billing, job_service as jobs, job_request_service as admission


class JobAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.stack = ExitStack()
        root = Path(self.temp.name)
        self.executor = ThreadPoolExecutor(max_workers=1)
        for target, name, value in [
            (jobs, "JOBS_DIR", root / "jobs"), (jobs, "JOBS_DB", root / "jobs/jobs.sqlite3"),
            (jobs, "_INITIALIZED", False), (jobs, "_EXECUTOR", self.executor),
            (billing, "BILLING_DIR", root / "billing"), (billing, "BILLING_DB", root / "billing/billing.sqlite3"),
            (billing, "_INITIALIZED", False), (settings, "billing_enforce_credits", True),
        ]:
            self.stack.enter_context(patch.object(target, name, value))
        jobs.initialize_job_store()
        billing._ledger("alice", 100, "test", "starting-credits")
        self.payload = VideoGenerationRequest(prompt="A presenter in a studio.", duration_seconds=15, request_id="one-take")

    def tearDown(self):
        self.executor.shutdown(wait=True)
        self.stack.close()
        self.temp.cleanup()

    def wait(self, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = jobs.get_job(job_id)
            if job["status"] in {"completed", "failed"}:
                return job
            time.sleep(.01)
        self.fail("Job did not finish")

    def test_duplicate_request_charges_and_renders_once(self):
        runner = Mock(return_value={"filename": "completed.mp4"})
        first = admission.submit_generation_request("video", self.payload, "alice", runner, charge_seconds=15)
        self.wait(first.job_id)
        second = admission.submit_generation_request("video", self.payload, "alice", runner, charge_seconds=15)
        self.assertEqual(first.job_id, second.job_id)
        self.assertEqual(billing.credit_balance("alice"), 85)
        runner.assert_called_once()
        with self.assertRaises(HTTPException) as error:
            admission.submit_generation_request("video", self.payload.model_copy(update={"prompt": "A completely different requested studio scene."}), "alice", runner, charge_seconds=15)
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(billing.credit_balance("alice"), 85)

    def test_queue_failure_refunds_and_retry_reserves_again(self):
        with patch.object(admission, "submit_job", side_effect=jobs.JobQueueFullError("Queue full")):
            with self.assertRaises(HTTPException):
                admission.submit_generation_request("video", self.payload, "alice", lambda _: {}, charge_seconds=15)
        self.assertEqual(billing.credit_balance("alice"), 100)
        response = admission.submit_generation_request("video", self.payload, "alice", lambda _: {}, charge_seconds=15)
        self.wait(response.job_id)
        self.assertEqual(billing.credit_balance("alice"), 85)

    def test_restart_refunds_charge_reserved_before_job_was_written(self):
        billing.consume_credits("alice", 15, "video:alice:orphan")
        self.assertEqual(billing.credit_balance("alice"), 85)
        jobs._INITIALIZED = False
        jobs.initialize_job_store()
        self.assertEqual(billing.credit_balance("alice"), 100)
        jobs._INITIALIZED = False
        jobs.initialize_job_store()
        self.assertEqual(billing.credit_balance("alice"), 100)

    def test_restart_refunds_interrupted_job_but_keeps_assets(self):
        reference = "video:alice:interrupted"
        billing.consume_credits("alice", 15, reference)
        job_id = jobs.create_job("video", {"workspace_id": "alice", "charge_reference": reference,
                                          "charge_seconds": 15, "credits_charged": True})
        jobs.update_job(job_id, status="running", result={"filename": "partial.mp4"})
        jobs._INITIALIZED = False
        jobs.initialize_job_store()
        self.assertEqual(jobs.get_job(job_id)["status"], "failed")
        self.assertEqual(billing.credit_balance("alice"), 100)
        self.assertTrue(jobs.workspace_owns_generated_file("alice", "partial.mp4"))

    def test_executor_rejection_refunds_and_marks_failed(self):
        rejected = Mock()
        rejected.submit.side_effect = RuntimeError("executor shut down")
        with patch.object(jobs, "_EXECUTOR", rejected):
            with self.assertRaises(RuntimeError):
                admission.submit_generation_request("video", self.payload, "alice", lambda _: {}, charge_seconds=15)
        self.assertEqual(billing.credit_balance("alice"), 100)
        self.assertEqual(jobs.find_job_by_request_id("alice", "one-take")["status"], "failed")

    def test_completed_pruned_job_cannot_be_refunded_on_restart(self):
        response = admission.submit_generation_request("video", self.payload, "alice", lambda _: {}, charge_seconds=15)
        self.wait(response.job_id)
        self.executor.shutdown(wait=True)
        with sqlite3.connect(jobs.JOBS_DB) as connection:
            connection.execute("DELETE FROM generation_jobs WHERE id=?", (response.job_id,))
        jobs._INITIALIZED = False
        jobs.initialize_job_store()
        self.assertEqual(billing.credit_balance("alice"), 85)


if __name__ == "__main__":
    unittest.main()
