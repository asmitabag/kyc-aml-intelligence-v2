"""SQLite helpers for the AML service (owner: Saloni)."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd


def default_db_path() -> Path:
    root = Path(os.environ.get("KYCAML_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    return Path(os.environ.get("KYCAML_DB", root / "db" / "kycaml.sqlite"))


def connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def next_id(conn: sqlite3.Connection, table: str, column: str, prefix: str, width: int = 6) -> str:
    """Next opaque ID like A000123. Call inside a write transaction (BEGIN IMMEDIATE)."""
    row = conn.execute(
        f"SELECT MAX(CAST(SUBSTR({column}, ?) AS INTEGER)) FROM {table} WHERE {column} LIKE ?",
        (len(prefix) + 1, prefix + "%")).fetchone()
    return f"{prefix}{(row[0] or 0) + 1:0{width}d}"


def load_transactions(conn) -> pd.DataFrame:
    """Transactions WITHOUT the label column: nothing downstream of this function can see labels."""
    df = pd.read_sql_query("SELECT transaction_id, from_account_id, to_account_id, amount, timestamp FROM transactions", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_accounts(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT account_id, customer_id, account_type, currency, created_at FROM accounts", conn)


def load_kyc(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT customer_id, kyc_risk_score, created_at FROM kyc_results", conn)


def signature(conn) -> tuple:
    """Cheap change detector for cached graph scores."""
    t = conn.execute("SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM transactions").fetchone()
    a = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()
    k = conn.execute("SELECT COUNT(*), COALESCE(MAX(rowid), 0) FROM kyc_results").fetchone()
    return (t[0], t[1], a[0], k[0], k[1])
