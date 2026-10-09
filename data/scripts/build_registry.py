"""Build the canonical registry: customers, accounts, transactions, labels, ID mappings (owner: Saloni).

Reads  data/interim/transactions_source.csv
Writes data/processed/{customers,accounts,transactions,aml_labels,id_mappings,documents,quarantine_transactions}.csv

Rules enforced here
  * Canonical IDs (C000001, A000001, T000001) are opaque. Accounts are numbered in order of a seeded hash of the
    source ID, so a canonical ID can never encode source row order or a label.
  * The AML sources contain accounts, not people. Customers are SYNTHETIC: each is assigned 1-3 accounts by a seeded
    process that never looks at labels. Every such link is recorded in id_mappings as SYNTHETIC_ASSOCIATION.
  * Rows with unresolved references, bad amounts, bad timestamps or non-ISO currencies are quarantined, not repaired.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (CURRENCY_RE, DEFAULT_SEED, INTERIM_DIR, PROCESSED_DIR, PROCESSED_TABLE_COLUMNS, ensure_dirs,
                    fmt_id, stable_hash)

ACCOUNT_LABEL_DEFINITION = ("1 if the account is sender or receiver of at least one transaction that the source "
                            "labels as suspicious/laundering; else 0. Derived from transaction labels only.")
TX_LABEL_DEFINITION = "Source label: transaction belongs to a known laundering/suspicious pattern (1) or not (0)."


def quarantine_reason(df: pd.DataFrame) -> pd.Series:
    reason = pd.Series("", index=df.index)

    def flag(mask, text):
        nonlocal reason
        reason = reason.where(~mask | (reason != ""), text)

    flag(df["source_tx_id"].isna() | df["source_from_account"].isna() | df["source_to_account"].isna(),
         "UNRESOLVED_REFERENCE")
    flag(df["source_tx_id"].duplicated(keep="first") & df["source_tx_id"].notna(), "DUPLICATE_SOURCE_TX_ID")
    flag(~(df["amount"] > 0) | ~np.isfinite(df["amount"].astype(float)), "INVALID_AMOUNT")
    flag(df["timestamp"].isna(), "INVALID_TIMESTAMP")
    flag(~df["currency"].fillna("").map(lambda c: bool(CURRENCY_RE.match(str(c)))), "INVALID_CURRENCY")
    flag(df["is_suspicious_label"].isna() | ~df["is_suspicious_label"].isin([0, 1]), "INVALID_LABEL")
    return reason


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    ensure_dirs()

    src = pd.read_csv(INTERIM_DIR / "transactions_source.csv", dtype={
        "source_name": str, "source_tx_id": str, "source_from_account": str, "source_to_account": str,
        "currency": str, "transaction_type": str})
    src["amount"] = pd.to_numeric(src["amount"], errors="coerce")
    src["is_suspicious_label"] = pd.to_numeric(src["is_suspicious_label"], errors="coerce")
    source_name = src["source_name"].iloc[0]
    data_origin = "SYNTHETIC" if source_name == "fixture" else "IMPORTED"

    reason = quarantine_reason(src)
    quarantined = src[reason != ""].assign(reason=reason[reason != ""])
    tx = src[reason == ""].copy()

    # ---- accounts: canonical IDs by seeded hash order --------------------------------------------
    accounts_src = sorted(set(tx["source_from_account"]) | set(tx["source_to_account"]),
                          key=lambda a: stable_hash("account:" + a, args.seed))
    acct_id = {a: fmt_id("A", i + 1) for i, a in enumerate(accounts_src)}

    # ---- synthetic customers: 1-3 accounts each, label-blind --------------------------------------
    rng = np.random.default_rng(args.seed)
    sizes_pool = np.array([1, 1, 1, 1, 2, 2, 3])
    cust_of: dict[str, str] = {}
    i = n_cust = 0
    while i < len(accounts_src):
        n_cust += 1
        size = int(rng.choice(sizes_pool))
        for a in accounts_src[i:i + size]:
            cust_of[a] = fmt_id("C", n_cust)
        i += size

    # ---- transactions ------------------------------------------------------------------------------
    tx["_t"] = pd.to_datetime(tx["timestamp"], utc=True)
    tx["_h"] = tx["source_tx_id"].map(lambda s: stable_hash("tx:" + s, args.seed))
    tx = tx.sort_values(["_t", "_h"]).reset_index(drop=True)
    tx["transaction_id"] = [fmt_id("T", i + 1) for i in range(len(tx))]
    tx["from_account_id"] = tx["source_from_account"].map(acct_id)
    tx["to_account_id"] = tx["source_to_account"].map(acct_id)
    tx["data_origin"] = data_origin
    tx["is_suspicious_label"] = tx["is_suspicious_label"].astype(int)
    transactions = tx[PROCESSED_TABLE_COLUMNS["transactions"]]

    # ---- accounts / customers tables --------------------------------------------------------------
    long = pd.concat([
        pd.DataFrame({"src": tx["source_from_account"], "t": tx["_t"], "cur": tx["currency"]}),
        pd.DataFrame({"src": tx["source_to_account"], "t": tx["_t"], "cur": tx["currency"]}),
    ])
    first_seen = long.groupby("src")["t"].min()
    modal_cur = long.groupby("src")["cur"].agg(lambda s: s.mode().iloc[0])
    acct_rng = np.random.default_rng(args.seed + 1)
    rows = []
    for a in accounts_src:
        rows.append({
            "account_id": acct_id[a], "customer_id": cust_of[a],
            "account_type": "SAVINGS" if acct_rng.random() < 0.7 else "CURRENT",
            "currency": modal_cur[a],
            "created_at": (first_seen[a] - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
    accounts = pd.DataFrame(rows, columns=PROCESSED_TABLE_COLUMNS["accounts"])
    cust_created = accounts.groupby("customer_id")["created_at"].min()
    customers = pd.DataFrame({
        "customer_id": cust_created.index,
        "full_name": [f"Synthetic Customer {c}" for c in cust_created.index],
        "date_of_birth": "",
        "created_at": cust_created.values,
        "customer_type": "INDIVIDUAL",
        "data_origin": "SYNTHETIC",
    })[PROCESSED_TABLE_COLUMNS["customers"]]

    # ---- labels ------------------------------------------------------------------------------------
    pos_accounts = set(transactions.loc[transactions["is_suspicious_label"] == 1, "from_account_id"]) | \
        set(transactions.loc[transactions["is_suspicious_label"] == 1, "to_account_id"])
    labels = pd.concat([
        pd.DataFrame({"entity_type": "TRANSACTION", "entity_id": transactions["transaction_id"],
                      "label": transactions["is_suspicious_label"], "label_definition": TX_LABEL_DEFINITION,
                      "provenance": source_name}),
        pd.DataFrame({"entity_type": "ACCOUNT", "entity_id": accounts["account_id"],
                      "label": accounts["account_id"].isin(pos_accounts).astype(int),
                      "label_definition": ACCOUNT_LABEL_DEFINITION, "provenance": source_name}),
    ], ignore_index=True)

    # ---- id mappings -------------------------------------------------------------------------------
    maps = [
        pd.DataFrame({"source_name": source_name, "source_entity_type": "ACCOUNT", "source_id": accounts_src,
                      "canonical_entity_type": "ACCOUNT", "canonical_id": [acct_id[a] for a in accounts_src],
                      "link_type": "SOURCE_IDENTITY", "notes": "source account preserved"}),
        pd.DataFrame({"source_name": source_name, "source_entity_type": "TRANSACTION",
                      "source_id": tx["source_tx_id"], "canonical_entity_type": "TRANSACTION",
                      "canonical_id": tx["transaction_id"], "link_type": "SOURCE_IDENTITY",
                      "notes": "source transaction preserved"}),
        pd.DataFrame({"source_name": source_name, "source_entity_type": "ACCOUNT", "source_id": accounts_src,
                      "canonical_entity_type": "CUSTOMER", "canonical_id": [cust_of[a] for a in accounts_src],
                      "link_type": "SYNTHETIC_ASSOCIATION",
                      "notes": "no real owner exists in the source; customer is synthetic and assigned by seeded process"}),
    ]
    mappings = pd.concat(maps, ignore_index=True)[PROCESSED_TABLE_COLUMNS["id_mappings"]]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_csv(PROCESSED_DIR / "customers.csv", index=False)
    accounts.to_csv(PROCESSED_DIR / "accounts.csv", index=False)
    transactions.to_csv(PROCESSED_DIR / "transactions.csv", index=False)
    labels.to_csv(PROCESSED_DIR / "aml_labels.csv", index=False)
    mappings.to_csv(PROCESSED_DIR / "id_mappings.csv", index=False)
    pd.DataFrame(columns=PROCESSED_TABLE_COLUMNS["documents"]).to_csv(PROCESSED_DIR / "documents.csv", index=False)
    quarantined.to_csv(PROCESSED_DIR / "quarantine_transactions.csv", index=False)

    print(f"customers={len(customers):,} accounts={len(accounts):,} transactions={len(transactions):,} "
          f"quarantined={len(quarantined):,} suspicious_accounts={len(pos_accounts):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
