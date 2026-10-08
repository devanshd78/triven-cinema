import hashlib
import hmac
import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import httpx

from app.core.config import settings


PROJECT_ROOT = Path(__file__).resolve().parents[4]
BILLING_DIR = PROJECT_ROOT / "storage" / "billing"
BILLING_DB = BILLING_DIR / "billing.sqlite3"
_DB_LOCK = Lock()
_INITIALIZED = False


class BillingError(RuntimeError):
    pass


class InsufficientCreditsError(BillingError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    BILLING_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(BILLING_DB, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_billing_store() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with _DB_LOCK:
        if _INITIALIZED:
            return
        with closing(_connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS billing_customers (
                    workspace_id TEXT PRIMARY KEY,
                    stripe_customer_id TEXT,
                    email TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS credit_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    delta_seconds INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    reference TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS stripe_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    processed_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_credit_workspace ON credit_ledger(workspace_id)"
            )
            connection.execute("""CREATE TABLE IF NOT EXISTS generation_reservations (
                reference TEXT PRIMARY KEY, workspace_id TEXT NOT NULL,
                seconds INTEGER NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL
            )""")
            connection.commit()
        _INITIALIZED = True


def ensure_customer(workspace_id: str) -> None:
    initialize_billing_store()
    timestamp = _now()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO billing_customers(workspace_id, created_at, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(workspace_id) DO NOTHING
            """,
            (workspace_id, timestamp, timestamp),
        )
        connection.commit()


def customer_record(workspace_id: str) -> dict:
    ensure_customer(workspace_id)
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT * FROM billing_customers WHERE workspace_id = ?", (workspace_id,)
        ).fetchone()
    return dict(row) if row else {"workspace_id": workspace_id}


def credit_balance(workspace_id: str) -> int:
    initialize_billing_store()
    with _DB_LOCK, closing(_connect()) as connection:
        row = connection.execute(
            "SELECT COALESCE(SUM(delta_seconds), 0) AS balance FROM credit_ledger WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
    return int(row["balance"] if row else 0)


def _ledger(workspace_id: str, delta_seconds: int, reason: str, reference: str) -> bool:
    initialize_billing_store()
    with _DB_LOCK, closing(_connect()) as connection:
        try:
            connection.execute(
                """
                INSERT INTO credit_ledger(workspace_id, delta_seconds, reason, reference, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (workspace_id, int(delta_seconds), reason, reference, _now()),
            )
            connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def _reservation(connection: sqlite3.Connection, workspace_id: str, reference: str):
    row = connection.execute("SELECT * FROM generation_reservations WHERE reference=?", (reference,)).fetchone()
    if row:
        if row["workspace_id"] != workspace_id:
            raise BillingError("Generation reservation belongs to another account.")
        return row
    # Existing ledgers remain usable through the reservation migration.
    debit = connection.execute(
        "SELECT * FROM credit_ledger WHERE reference=? AND workspace_id=? AND delta_seconds < 0",
        (reference, workspace_id),
    ).fetchone()
    if debit:
        refunded = connection.execute("SELECT 1 FROM credit_ledger WHERE reference=?", (f"refund:{reference}",)).fetchone()
        connection.execute("INSERT INTO generation_reservations VALUES(?,?,?,?,0)",
                           (reference, workspace_id, -debit["delta_seconds"], "refunded" if refunded else "charged"))
        return connection.execute("SELECT * FROM generation_reservations WHERE reference=?", (reference,)).fetchone()
    return None


def consume_credits(workspace_id: str, seconds: int, reference: str) -> None:
    if not settings.billing_enforce_credits:
        return
    required = max(1, int(seconds))
    initialize_billing_store()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = _reservation(connection, workspace_id, reference)
        if existing and existing["state"] == "settled":
            raise BillingError("This generation request has already completed. Start a new take.")
        if existing and existing["state"] == "charged":
            if existing["seconds"] != required:
                raise BillingError("This generation request already reserved a different duration.")
            connection.commit()
            return
        balance = connection.execute(
            "SELECT COALESCE(SUM(delta_seconds),0) FROM credit_ledger WHERE workspace_id=?", (workspace_id,)
        ).fetchone()[0]
        if balance < required:
            raise InsufficientCreditsError(f"Insufficient generation credits. Need {required}s; balance is {balance}s.")
        attempt = int(existing["attempt"]) + 1 if existing else 0
        ledger_ref = reference if attempt == 0 else f"retry:{reference}:{attempt}"
        connection.execute(
            "INSERT INTO credit_ledger(workspace_id,delta_seconds,reason,reference,created_at) VALUES(?,?,'generation',?,?)",
            (workspace_id, -required, ledger_ref, _now()),
        )
        connection.execute(
            """INSERT INTO generation_reservations VALUES(?,?,?,'charged',?)
            ON CONFLICT(reference) DO UPDATE SET seconds=excluded.seconds,state='charged',attempt=excluded.attempt""",
            (reference, workspace_id, required, attempt),
        )
        connection.commit()


def refund_credits(workspace_id: str, seconds: int, reference: str) -> None:
    # Refund the recorded charge, never a caller-supplied amount or an uncharged
    # request. Repeated failure callbacks and restart recovery are idempotent.
    initialize_billing_store()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        reservation = _reservation(connection, workspace_id, reference)
        if not reservation or reservation["state"] != "charged":
            connection.commit()
            return
        attempt = int(reservation["attempt"])
        ledger_ref = f"refund:{reference}" if attempt == 0 else f"refund:{reference}:{attempt}"
        connection.execute(
            "INSERT OR IGNORE INTO credit_ledger(workspace_id,delta_seconds,reason,reference,created_at) VALUES(?,?,'generation_refund',?,?)",
            (workspace_id, reservation["seconds"], ledger_ref, _now()),
        )
        connection.execute("UPDATE generation_reservations SET state='refunded' WHERE reference=?", (reference,))
        connection.commit()


def settle_credits(workspace_id: str, reference: str) -> None:
    initialize_billing_store()
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            "UPDATE generation_reservations SET state='settled' WHERE reference=? AND workspace_id=? AND state='charged'",
            (reference, workspace_id),
        )
        connection.commit()


def reconcile_orphaned_generation_charges(known_references: set[str]) -> None:
    """Recover a crash between reserving credits and persisting the job row.

    Called before request admission on the single-process application startup.
    Only the new reservation records are reconciled; legacy completed debits
    without reservation metadata must never be inferred to be orphaned.
    """
    initialize_billing_store()
    with _DB_LOCK, closing(_connect()) as connection:
        rows = connection.execute("SELECT * FROM generation_reservations WHERE state='charged'").fetchall()
    for row in rows:
        if row["reference"] not in known_references:
            refund_credits(row["workspace_id"], row["seconds"], row["reference"])


def catalog() -> list[dict]:
    return [
        {
            "id": "starter",
            "label": "Starter · 5 generation minutes",
            "credit_seconds": int(settings.stripe_starter_seconds),
            "price_id": settings.stripe_price_starter.strip(),
        },
        {
            "id": "pro",
            "label": "Pro · 30 generation minutes",
            "credit_seconds": int(settings.stripe_pro_seconds),
            "price_id": settings.stripe_price_pro.strip(),
        },
        {
            "id": "studio",
            "label": "Studio · 120 generation minutes",
            "credit_seconds": int(settings.stripe_studio_seconds),
            "price_id": settings.stripe_price_studio.strip(),
        },
    ]


def _pack(pack_id: str) -> dict:
    item = next((item for item in catalog() if item["id"] == pack_id), None)
    if not item:
        raise BillingError("Unknown credit pack.")
    if not item["price_id"]:
        raise BillingError(f"Stripe price for {pack_id} is not configured.")
    return item


def _stripe_request(method: str, path: str, *, data: dict | None = None) -> dict:
    secret = settings.stripe_secret_key.strip()
    if not secret:
        raise BillingError("STRIPE_SECRET_KEY is not configured.")
    url = f"https://api.stripe.com/v1{path}"
    try:
        response = httpx.request(
            method,
            url,
            auth=(secret, ""),
            data=data,
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise BillingError("Stripe request failed.") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise BillingError("Stripe returned an invalid response.") from exc
    if response.status_code >= 400:
        message = ((payload.get("error") or {}).get("message") or "Stripe request failed.")
        raise BillingError(str(message))
    return payload


def create_checkout_session(workspace_id: str, pack_id: str) -> dict:
    if not settings.billing_enabled:
        raise BillingError("Billing is disabled.")
    pack = _pack(pack_id)
    customer = customer_record(workspace_id)
    base = settings.frontend_url.rstrip("/")
    data = {
        "mode": "payment",
        "line_items[0][price]": pack["price_id"],
        "line_items[0][quantity]": "1",
        "success_url": f"{base}/?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{base}/?checkout=cancelled",
        "client_reference_id": workspace_id,
        "metadata[workspace_id]": workspace_id,
        "metadata[credit_seconds]": str(pack["credit_seconds"]),
        "metadata[pack_id]": pack["id"],
        "allow_promotion_codes": "true",
    }
    if customer.get("stripe_customer_id"):
        data["customer"] = customer["stripe_customer_id"]
    else:
        data["customer_creation"] = "always"
    session = _stripe_request("POST", "/checkout/sessions", data=data)
    if not session.get("url"):
        raise BillingError("Stripe Checkout did not return a URL.")
    return session


def _record_customer(workspace_id: str, session: dict) -> None:
    ensure_customer(workspace_id)
    customer_details = session.get("customer_details") or {}
    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            """
            UPDATE billing_customers
            SET stripe_customer_id = COALESCE(?, stripe_customer_id),
                email = COALESCE(?, email),
                updated_at = ?
            WHERE workspace_id = ?
            """,
            (
                session.get("customer"),
                customer_details.get("email"),
                _now(),
                workspace_id,
            ),
        )
        connection.commit()


def process_checkout_session(session: dict) -> bool:
    metadata = session.get("metadata") or {}
    workspace_id = str(metadata.get("workspace_id") or session.get("client_reference_id") or "")
    if not workspace_id:
        raise BillingError("Checkout session is missing workspace metadata.")
    payment_status = str(session.get("payment_status") or "")
    if payment_status not in {"paid", "no_payment_required"}:
        return False
    try:
        credit_seconds = int(metadata.get("credit_seconds") or 0)
    except (TypeError, ValueError):
        credit_seconds = 0
    if credit_seconds <= 0:
        raise BillingError("Checkout session has invalid credit metadata.")

    _record_customer(workspace_id, session)
    return _ledger(
        workspace_id,
        credit_seconds,
        "stripe_checkout",
        f"checkout:{session.get('id')}",
    )


def verify_checkout_session(workspace_id: str, session_id: str) -> dict:
    session = _stripe_request("GET", f"/checkout/sessions/{session_id}")
    session_workspace = str(
        (session.get("metadata") or {}).get("workspace_id")
        or session.get("client_reference_id")
        or ""
    )
    if session_workspace != workspace_id:
        raise BillingError("Checkout session does not belong to this workspace.")
    process_checkout_session(session)
    return session


def create_billing_portal(workspace_id: str) -> str:
    customer = customer_record(workspace_id)
    stripe_customer_id = customer.get("stripe_customer_id")
    if not stripe_customer_id:
        raise BillingError("No Stripe customer exists for this workspace yet.")
    payload = _stripe_request(
        "POST",
        "/billing_portal/sessions",
        data={
            "customer": stripe_customer_id,
            "return_url": settings.frontend_url.rstrip("/"),
        },
    )
    url = payload.get("url")
    if not url:
        raise BillingError("Stripe Billing Portal did not return a URL.")
    return str(url)


def verify_stripe_signature(payload: bytes, signature_header: str) -> None:
    secret = settings.stripe_webhook_secret.strip()
    if not secret:
        raise BillingError("STRIPE_WEBHOOK_SECRET is not configured.")
    parts: dict[str, list[str]] = {}
    for item in signature_header.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts.setdefault(key.strip(), []).append(value.strip())
    timestamp = (parts.get("t") or [""])[0]
    signatures = parts.get("v1") or []
    if not timestamp or not signatures:
        raise BillingError("Invalid Stripe-Signature header.")
    try:
        ts = int(timestamp)
    except ValueError as exc:
        raise BillingError("Invalid Stripe webhook timestamp.") from exc
    if abs(int(time.time()) - ts) > 300:
        raise BillingError("Stripe webhook timestamp is outside the allowed tolerance.")
    signed_payload = timestamp.encode("utf-8") + b"." + payload
    expected = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, supplied) for supplied in signatures):
        raise BillingError("Stripe webhook signature verification failed.")


def process_webhook(payload: bytes, signature_header: str) -> str:
    verify_stripe_signature(payload, signature_header)
    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BillingError("Invalid Stripe webhook JSON.") from exc
    event_id = str(event.get("id") or "")
    event_type = str(event.get("type") or "")
    if not event_id:
        raise BillingError("Stripe event is missing an id.")

    initialize_billing_store()
    with _DB_LOCK, closing(_connect()) as connection:
        if connection.execute("SELECT 1 FROM stripe_events WHERE event_id = ?", (event_id,)).fetchone():
            return event_type

    if event_type in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
        process_checkout_session((event.get("data") or {}).get("object") or {})

    with _DB_LOCK, closing(_connect()) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO stripe_events(event_id, event_type, processed_at) VALUES (?, ?, ?)",
            (event_id, event_type, _now()),
        )
        connection.commit()
    return event_type


initialize_billing_store()
