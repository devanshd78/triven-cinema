import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.core.config import settings
from app.services import billing_service as billing


class CreditReservationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        for obj,name,value in [(billing,'BILLING_DIR',root),(billing,'BILLING_DB',root/'billing.sqlite3'),(billing,'_INITIALIZED',False),(settings,'billing_enforce_credits',True)]:
            self.stack.enter_context(patch.object(obj,name,value))
        billing._ledger('alice',100,'test-credit','deposit')

    def test_duplicate_charge_refund_and_recharged_retry_are_idempotent(self):
        billing.consume_credits('alice',15,'request-one')
        billing.consume_credits('alice',15,'request-one')
        self.assertEqual(billing.credit_balance('alice'),85)
        billing.refund_credits('alice',999,'request-one')
        billing.refund_credits('alice',999,'request-one')
        self.assertEqual(billing.credit_balance('alice'),100)
        billing.consume_credits('alice',15,'request-one')
        self.assertEqual(billing.credit_balance('alice'),85)
        billing.refund_credits('alice',15,'request-one')
        self.assertEqual(billing.credit_balance('alice'),100)

    def test_cannot_refund_unpaid_or_another_accounts_request(self):
        billing.refund_credits('alice',999,'never-charged')
        self.assertEqual(billing.credit_balance('alice'),100)
        billing.consume_credits('alice',15,'request-one')
        with self.assertRaises(billing.BillingError):
            billing.refund_credits('bob',15,'request-one')
        self.assertEqual(billing.credit_balance('bob'),0)

    def test_existing_charge_can_be_refunded_after_enforcement_disabled(self):
        billing.consume_credits('alice',15,'request-one')
        with patch.object(settings,'billing_enforce_credits',False):
            billing.refund_credits('alice',15,'request-one')
        self.assertEqual(billing.credit_balance('alice'),100)


if __name__ == '__main__':
    unittest.main()
