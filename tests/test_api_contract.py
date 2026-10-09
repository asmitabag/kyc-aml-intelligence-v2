import sqlite3
import unittest

try:
    from fastapi.testclient import TestClient
    HAVE_CLIENT = True
except Exception:  # httpx missing
    HAVE_CLIENT = False

from tests.support import fresh_db_copy


class StubGnn:
    status, reason = "loaded", ""

    def score_account(self, account_id, include_evidence=True):
        if account_id != "A000001":
            raise KeyError(account_id)
        return {"account_id": account_id, "gnn_risk_score": 0.5}


@unittest.skipUnless(HAVE_CLIENT, "fastapi TestClient needs httpx")
class ApiContractTests(unittest.TestCase):
    def setUp(self):
        from services.aml.api import create_app
        self.db = fresh_db_copy()
        self.client = TestClient(create_app(self.db, StubGnn()), raise_server_exceptions=False)

    def assert_error(self, resp, status, code):
        self.assertEqual(resp.status_code, status, resp.text)
        err = resp.json()["error"]
        self.assertEqual(set(err), {"code", "message", "request_id"})
        if code:
            self.assertEqual(err["code"], code)

    def test_create_account(self):
        r = self.client.post("/api/v1/accounts", json={"customer_id": "C000001", "account_type": "SAVINGS", "currency": "INR"})
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(set(body), {"account_id", "customer_id", "account_type", "currency"})
        self.assertRegex(body["account_id"], r"^A\d{6}$")
        again = self.client.post("/api/v1/accounts", json={"customer_id": "C000001", "account_type": "CURRENT", "currency": "INR"})
        self.assertNotEqual(again.json()["account_id"], body["account_id"])

    def test_create_account_errors(self):
        self.assert_error(self.client.post("/api/v1/accounts", json={"customer_id": "NOPE", "account_type": "SAVINGS", "currency": "INR"}),
                          404, "CUSTOMER_NOT_FOUND")
        self.assert_error(self.client.post("/api/v1/accounts", json={"customer_id": "C000001", "account_type": "SAVINGS", "currency": "rupee"}),
                          422, "VALIDATION_ERROR")

    def _tx(self, tid, **over):
        row = {"transaction_id": tid, "from_account_id": "A000001", "to_account_id": "A000002", "amount": 45000.0,
               "currency": "INR", "timestamp": "2026-10-09T10:00:00Z", "transaction_type": "TRANSFER", "data_origin": "SYNTHETIC"}
        row.update(over)
        return row

    def test_import_accepts_valid_and_rejects_invalid_rows(self):
        rows = [self._tx("TNEW1"),
                self._tx("TNEW2", is_suspicious_label=1),
                self._tx("TNEW3", to_account_id="A999999"),
                self._tx("TNEW4", amount=0),
                self._tx("TNEW5", timestamp="2026-10-09T10:00:00"),
                self._tx("TNEW6", currency="inr"),
                self._tx("T000001"),
                self._tx("TNEW1")]
        r = self.client.post("/api/v1/transactions/import", json={"transactions": rows})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["imported_count"], 1)
        self.assertEqual(body["rejected_count"], 7)
        codes = {e["code"] for e in body["errors"]}
        self.assertTrue({"LABEL_NOT_ALLOWED", "ACCOUNT_NOT_FOUND", "INVALID_AMOUNT", "INVALID_TIMESTAMP",
                         "INVALID_CURRENCY", "DUPLICATE_TRANSACTION_ID"} <= codes, codes)
        conn = sqlite3.connect(self.db)
        label = conn.execute("SELECT is_suspicious_label FROM transactions WHERE transaction_id='TNEW1'").fetchone()[0]
        self.assertIsNone(label)  # imported rows carry no label

    def test_customer_transactions_include_import_and_hide_labels(self):
        conn = sqlite3.connect(self.db)
        cust = conn.execute("SELECT customer_id FROM accounts WHERE account_id='A000001'").fetchone()[0]
        self.client.post("/api/v1/transactions/import", json={"transactions": [self._tx("TNEW9")]})
        r = self.client.get(f"/api/v1/customers/{cust}/transactions")
        self.assertEqual(r.status_code, 200)
        txs = r.json()["transactions"]
        self.assertIn("TNEW9", {t["transaction_id"] for t in txs})
        self.assertTrue(all("is_suspicious_label" not in t for t in txs))
        self.assert_error(self.client.get("/api/v1/customers/NOPE/transactions"), 404, "CUSTOMER_NOT_FOUND")

    def test_internal_gnn_route(self):
        self.assertEqual(self.client.post("/api/v1/internal/gnn-score", json={"account_id": "A000001"}).status_code, 200)
        self.assert_error(self.client.post("/api/v1/internal/gnn-score", json={"account_id": "A777777"}), 404, "ACCOUNT_NOT_FOUND")
        from services.aml.api import create_app
        no_model = TestClient(create_app(self.db, None), raise_server_exceptions=False)
        self.assert_error(no_model.post("/api/v1/internal/gnn-score", json={"account_id": "A000001"}), 503, "MODEL_UNAVAILABLE")
        self.assertEqual(no_model.get("/api/v1/health").json()["models"]["gnn"], "unavailable")


class EvidenceTests(unittest.TestCase):
    def test_detects_cycle_rapid_movement_and_fan_out(self):
        import pandas as pd
        from services.aml.evidence import graph_evidence
        t0 = pd.Timestamp("2026-08-01T00:00:00Z")
        rows = [("T1", "A", "B", 1000, 0), ("T2", "B", "C", 970, 30), ("T3", "C", "A", 940, 60)]
        rows += [(f"F{i}", "A", f"X{i}", 50, 100 + i) for i in range(6)]
        tx = pd.DataFrame([{"transaction_id": a, "from_account_id": b, "to_account_id": c, "amount": float(d),
                            "timestamp": t0 + pd.Timedelta(minutes=m)} for a, b, c, d, m in rows])
        codes = {f["code"] for f in graph_evidence("A", tx)["flags"]}
        self.assertIn("CIRCULAR_FLOW", codes)
        self.assertIn("FAN_OUT", codes)
        self.assertIn("RAPID_FUND_MOVEMENT", {f["code"] for f in graph_evidence("B", tx)["flags"]})
        quiet = tx[tx["transaction_id"].isin(["T1"])]
        self.assertEqual(graph_evidence("A", quiet)["flags"], [])


if __name__ == "__main__":
    unittest.main()
