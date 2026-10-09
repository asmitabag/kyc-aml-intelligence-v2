"""Convert a transaction source into the interim canonical-source format (owner: Saloni).

    python data/scripts/prepare_transactions.py --source fixture
    python data/scripts/prepare_transactions.py --source ibm_aml --input data/raw/ibm_aml/HI-Small_Trans.csv
    python data/scripts/prepare_transactions.py --source amlsim  --input data/raw/amlsim/transactions.csv

Output: data/interim/transactions_source.csv with source-level IDs preserved. build_registry.py then assigns the
canonical IDs. CHECK THE COLUMN NAMES against the file you actually downloaded: dataset versions differ, and the
loaders stop with a clear error rather than guessing.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import INTERIM_DIR, ensure_dirs
from make_fixture import CANONICAL_COLS, generate_fixture

# IBM AML publishes currency names, not ISO codes. Unmapped names are quarantined later, not guessed.
IBM_CURRENCY = {
    "US Dollar": "USD", "Euro": "EUR", "Yuan": "CNY", "Rupee": "INR", "Yen": "JPY", "UK Pound": "GBP",
    "Canadian Dollar": "CAD", "Swiss Franc": "CHF", "Australian Dollar": "AUD", "Ruble": "RUB",
    "Brazil Real": "BRL", "Mexican Peso": "MXN", "Saudi Riyal": "SAR", "Shekel": "ILS", "Bitcoin": "BTC",
}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.copy()
    df["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # NaT -> NaN, quarantined downstream
    return df[CANONICAL_COLS]


def load_ibm_aml(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"From Bank": str, "To Bank": str, "Account": str, "Account.1": str})
    need = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Paid", "Payment Currency",
            "Payment Format", "Is Laundering"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise SystemExit(f"IBM AML file is missing columns {missing}. Found: {list(raw.columns)}")
    out = pd.DataFrame({
        "source_name": "ibm_aml",
        "source_tx_id": [f"IBM{i:09d}" for i in range(len(raw))],  # file has no tx ID: row number, documented
        "source_from_account": raw["From Bank"] + "_" + raw["Account"],  # an account = (bank, account number)
        "source_to_account": raw["To Bank"] + "_" + raw["Account.1"],
        "amount": raw["Amount Paid"],
        "currency": raw["Payment Currency"].map(IBM_CURRENCY),
        # Assumption: the file's timestamps carry no zone; treated as UTC.
        "timestamp": pd.to_datetime(raw["Timestamp"], errors="coerce", utc=True),
        "transaction_type": raw["Payment Format"].str.upper().str.replace(" ", "_"),
        "is_suspicious_label": raw["Is Laundering"],
    })
    return _finalize(out)


def load_amlsim(path: str, default_currency: str, step_hours: float) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    cols = {c.lower(): c for c in raw.columns}

    def pick(*options):
        for o in options:
            if o in cols:
                return cols[o]
        raise SystemExit(f"AMLSim file: none of {options} found. Columns present: {list(raw.columns)}")

    c_id = pick("tran_id", "tx_id", "transaction_id")
    c_src = pick("orig_acct", "sender_account_id", "sender", "orig_account")
    c_dst = pick("bene_acct", "receiver_account_id", "receiver", "bene_account")
    c_amt = pick("base_amt", "tx_amount", "amount")
    c_type = pick("tx_type", "type")
    c_ts = pick("tran_timestamp", "timestamp", "tx_step", "step")
    c_lab = pick("is_sar", "is_fraud", "is_laundering")
    ts_raw = raw[c_ts]
    if pd.to_numeric(ts_raw, errors="coerce").notna().all():  # simulation step -> assumed offset from a fixed date
        ts = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(pd.to_numeric(ts_raw) * step_hours, unit="h")
    else:
        ts = pd.to_datetime(ts_raw, errors="coerce", utc=True)
    lab = raw[c_lab].str.lower().map({"1": 1, "true": 1, "0": 0, "false": 0})
    out = pd.DataFrame({
        "source_name": "amlsim", "source_tx_id": raw[c_id], "source_from_account": raw[c_src],
        "source_to_account": raw[c_dst], "amount": pd.to_numeric(raw[c_amt], errors="coerce"),
        "currency": default_currency, "timestamp": ts, "transaction_type": raw[c_type].str.upper(),
        "is_suspicious_label": lab,
    })
    return _finalize(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fixture", "ibm_aml", "amlsim"], default="fixture")
    ap.add_argument("--input", help="path to the source CSV (not needed for fixture)")
    ap.add_argument("--default-currency", default="USD", help="AMLSim has no currency column (assumption)")
    ap.add_argument("--step-hours", type=float, default=24.0, help="AMLSim numeric step length (assumption)")
    ap.add_argument("--fixture-accounts", type=int, default=3000)
    ap.add_argument("--fixture-transactions", type=int, default=30000)
    ap.add_argument("--fixture-patterns", type=int, default=70)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    ensure_dirs()

    if args.source == "fixture":
        kw = {} if args.seed is None else {"seed": args.seed}
        df = generate_fixture(args.fixture_accounts, args.fixture_transactions, args.fixture_patterns, **kw)
    else:
        if not args.input:
            raise SystemExit("--input is required for this source")
        df = load_ibm_aml(args.input) if args.source == "ibm_aml" else load_amlsim(
            args.input, args.default_currency, args.step_hours)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / "transactions_source.csv"
    df.to_csv(out, index=False)
    pos = pd.to_numeric(df["is_suspicious_label"], errors="coerce").sum()
    print(f"{args.source}: wrote {len(df):,} rows -> {out}  (labelled suspicious: {int(pos):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
