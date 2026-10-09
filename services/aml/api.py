"""Saloni's API surface, following docs/API_CONTRACTS_V1.md (owner: Saloni).

  POST /api/v1/accounts                         API 3  register an account
  POST /api/v1/transactions/import              API 4  validated bulk import (rejects label fields)
  GET  /api/v1/customers/{id}/transactions      API 4  all transactions of a customer's accounts
  POST /api/v1/internal/gnn-score               PROPOSED internal route: GNN score + evidence for one account.
                                                Sounak's aml-analysis orchestrator (API 5) calls it. Not in the frozen
                                                contract yet: needs team approval before it is treated as frozen.

Sounak mounts it in the main app:
    from services.aml.api import create_router, register_error_handlers
    app.include_router(create_router(db_path, gnn_service)); register_error_handlers(app)
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from services.aml import db as DB

API_PREFIX = "/api/v1"
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
ALLOWED_TX_ORIGINS = {"SYNTHETIC", "IMPORTED"}
TX_FIELDS = {"transaction_id", "from_account_id", "to_account_id", "amount", "currency", "timestamp",
             "transaction_type", "data_origin"}
MAX_IMPORT_ROWS = 10_000


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(status_code=status, detail={"code": code, "message": message})


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"


def register_error_handlers(app: FastAPI) -> None:
    """Every error uses the contract's {"error": {code, message, request_id}} structure."""

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        d = exc.detail if isinstance(exc.detail, dict) and "code" in exc.detail else {"code": "HTTP_ERROR", "message": str(exc.detail)}
        return JSONResponse(status_code=exc.status_code, content={"error": {**d, "request_id": _request_id(request)}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
        return JSONResponse(status_code=422, content={"error": {
            "code": "VALIDATION_ERROR", "message": msg, "request_id": _request_id(request)}})

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"error": {
            "code": "INTERNAL_ERROR", "message": "An unexpected error occurred.", "request_id": _request_id(request)}})


class AccountCreate(BaseModel):
    customer_id: str = Field(min_length=1)
    account_type: str = Field(min_length=1, max_length=32)
    currency: str

    @field_validator("currency")
    @classmethod
    def _cur(cls, v: str) -> str:
        if not CURRENCY_RE.match(v):
            raise ValueError("currency must be a 3-letter uppercase ISO code, e.g. INR")
        return v


class TransactionImport(BaseModel):
    transactions: list[dict] = Field(min_length=1)


class GnnScoreRequest(BaseModel):
    account_id: str
    include_evidence: bool = True


def parse_transaction(row: dict) -> tuple[dict | None, dict | None]:
    """Validate one import row. Returns (clean_row, None) or (None, error)."""
    tid = row.get("transaction_id")

    def bad(code, msg):
        return None, {"transaction_id": tid, "code": code, "message": msg}

    if "is_suspicious_label" in row:
        return bad("LABEL_NOT_ALLOWED", "is_suspicious_label must not be supplied: labels are never model inputs")
    extra = set(row) - TX_FIELDS
    if extra:
        return bad("UNKNOWN_FIELD", f"unknown field(s): {sorted(extra)}")
    missing = [f for f in TX_FIELDS if row.get(f) in (None, "")]
    if missing:
        return bad("MISSING_FIELD", f"missing field(s): {sorted(missing)}")
    if not isinstance(tid, str):
        return bad("INVALID_TRANSACTION_ID", "transaction_id must be a string")
    try:
        amt = Decimal(str(row["amount"]))
        if not amt.is_finite() or amt <= 0:
            raise InvalidOperation
    except (InvalidOperation, ValueError):
        return bad("INVALID_AMOUNT", "amount must be a finite number greater than zero")
    if not CURRENCY_RE.match(str(row["currency"])):
        return bad("INVALID_CURRENCY", "currency must be a 3-letter uppercase ISO code")
    if row["data_origin"] not in ALLOWED_TX_ORIGINS:
        return bad("INVALID_DATA_ORIGIN", f"data_origin must be one of {sorted(ALLOWED_TX_ORIGINS)}")
    try:
        ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
    except ValueError:
        return bad("INVALID_TIMESTAMP", "timestamp must be ISO 8601")
    if ts.tzinfo is None:
        return bad("INVALID_TIMESTAMP", "timestamp must include a UTC offset (use 'Z')")
    if row["from_account_id"] == row["to_account_id"]:
        return bad("SELF_TRANSFER", "from_account_id and to_account_id must differ")
    clean = {**{k: str(row[k]) for k in ("transaction_id", "from_account_id", "to_account_id", "currency",
                                           "transaction_type", "data_origin")},
             "amount": float(amt), "timestamp": ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return clean, None


def create_router(db_path, gnn_service=None) -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)

    @router.post("/accounts", status_code=201)
    def create_account(body: AccountCreate):
        conn = DB.connect(db_path)
        try:
            if conn.execute("SELECT 1 FROM customers WHERE customer_id = ?", (body.customer_id,)).fetchone() is None:
                raise ApiError(404, "CUSTOMER_NOT_FOUND", "The requested customer does not exist.")
            conn.execute("BEGIN IMMEDIATE")
            account_id = DB.next_id(conn, "accounts", "account_id", "A")
            conn.execute("INSERT INTO accounts (account_id, customer_id, account_type, currency, created_at) VALUES (?,?,?,?,?)",
                         (account_id, body.customer_id, body.account_type, body.currency,
                          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            conn.commit()
            return {"account_id": account_id, "customer_id": body.customer_id,
                    "account_type": body.account_type, "currency": body.currency}
        finally:
            conn.close()

    @router.post("/transactions/import")
    def import_transactions(body: TransactionImport):
        if len(body.transactions) > MAX_IMPORT_ROWS:
            raise ApiError(400, "TOO_MANY_ROWS", f"at most {MAX_IMPORT_ROWS} transactions per request")
        conn = DB.connect(db_path)
        try:
            clean, errors = [], []
            seen = set()
            for row in body.transactions:
                ok, err = parse_transaction(row if isinstance(row, dict) else {})
                if err:
                    errors.append(err)
                elif ok["transaction_id"] in seen:
                    errors.append({"transaction_id": ok["transaction_id"], "code": "DUPLICATE_TRANSACTION_ID",
                                   "message": "transaction_id repeated within this request"})
                else:
                    seen.add(ok["transaction_id"])
                    clean.append(ok)
            if clean:
                ids = [r["transaction_id"] for r in clean]
                existing = {r[0] for r in conn.execute(
                    f"SELECT transaction_id FROM transactions WHERE transaction_id IN ({','.join('?' * len(ids))})", ids)}
                accts = {a for r in clean for a in (r["from_account_id"], r["to_account_id"])}
                known = {r[0] for r in conn.execute(
                    f"SELECT account_id FROM accounts WHERE account_id IN ({','.join('?' * len(accts))})", list(accts))}
                good = []
                for r in clean:
                    if r["transaction_id"] in existing:
                        errors.append({"transaction_id": r["transaction_id"], "code": "DUPLICATE_TRANSACTION_ID",
                                       "message": "transaction_id already exists"})
                    elif r["from_account_id"] not in known or r["to_account_id"] not in known:
                        errors.append({"transaction_id": r["transaction_id"], "code": "ACCOUNT_NOT_FOUND",
                                       "message": "from_account_id or to_account_id does not exist"})
                    else:
                        good.append(r)
                conn.execute("BEGIN IMMEDIATE")
                conn.executemany(
                    "INSERT INTO transactions (transaction_id, from_account_id, to_account_id, amount, currency, timestamp,"
                    " transaction_type, data_origin, is_suspicious_label) VALUES (?,?,?,?,?,?,?,?,NULL)",
                    [(r["transaction_id"], r["from_account_id"], r["to_account_id"], r["amount"], r["currency"],
                      r["timestamp"], r["transaction_type"], r["data_origin"]) for r in good])
                conn.commit()
                clean = good
            return {"imported_count": len(clean), "rejected_count": len(errors), "errors": errors}
        finally:
            conn.close()

    @router.get("/customers/{customer_id}/transactions")
    def customer_transactions(customer_id: str):
        conn = DB.connect(db_path)
        try:
            if conn.execute("SELECT 1 FROM customers WHERE customer_id = ?", (customer_id,)).fetchone() is None:
                raise ApiError(404, "CUSTOMER_NOT_FOUND", "The requested customer does not exist.")
            rows = conn.execute(
                "SELECT t.transaction_id, t.from_account_id, t.to_account_id, t.amount, t.currency, t.timestamp, "
                "t.transaction_type, t.data_origin FROM transactions t "
                "WHERE t.from_account_id IN (SELECT account_id FROM accounts WHERE customer_id = ?) "
                "   OR t.to_account_id   IN (SELECT account_id FROM accounts WHERE customer_id = ?) "
                "ORDER BY t.timestamp, t.transaction_id", (customer_id, customer_id)).fetchall()
            return {"customer_id": customer_id, "transactions": [dict(r) for r in rows]}  # labels are never returned
        finally:
            conn.close()

    @router.post("/internal/gnn-score")
    def gnn_score(body: GnnScoreRequest):
        if gnn_service is None or gnn_service.status != "loaded":
            reason = getattr(gnn_service, "reason", "GNN service not configured")
            raise ApiError(503, "MODEL_UNAVAILABLE", f"GNN model is unavailable: {reason}")
        try:
            return gnn_service.score_account(body.account_id, body.include_evidence)
        except KeyError:
            raise ApiError(404, "ACCOUNT_NOT_FOUND", "The requested account does not exist.")

    return router


def create_app(db_path=None, gnn_service=None) -> FastAPI:
    app = FastAPI(title="KYC-AML: dataset and GNN service (Saloni)", version="v1")
    register_error_handlers(app)
    app.include_router(create_router(db_path or DB.default_db_path(), gnn_service))

    @app.get(f"{API_PREFIX}/health")
    def health():
        return {"status": "ok", "api_version": "v1",
                "models": {"gnn": getattr(gnn_service, "status", "unavailable")}}

    return app
