"""Synthetic AMLSim-style fixture generator (owner: Saloni).

Purpose: give Asmita and Sounak a small, schema-compatible dataset immediately, and let the whole
pipeline + GNN be tested end-to-end before the real IBM AML / AMLSim data is available.

HONESTY NOTE: everything produced here is SYNTHETIC. Metrics measured on this fixture only show that the
code works. They are NOT research results and must never be reported as such.

Laundering patterns injected (each transaction in a pattern gets is_suspicious_label = 1):
  * cycle           - funds go A -> B -> C -> ... -> A with a small loss per hop
  * scatter_gather  - one source splits funds across mules that all forward to one sink
  * chain           - rapid layering A -> B -> C -> D ... with decaying amounts
Benign confounders (label 0) so the task is not trivial: merchants (large fan-in), payroll accounts
(large fan-out bursts) and ordinary rapid pass-through pairs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEFAULT_SEED

CANONICAL_COLS = [
    "source_name", "source_tx_id", "source_from_account", "source_to_account", "amount",
    "currency", "timestamp", "transaction_type", "is_suspicious_label",
]
BASE_TIME = pd.Timestamp("2026-07-01T00:00:00Z")
HORIZON_DAYS = 90


def generate_fixture(n_accounts: int = 3000, n_normal_tx: int = 30000, n_patterns: int = 70,
                     seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_accounts)  # source IDs carry no information about role or label
    names = [f"FX{p:06d}" for p in perm]
    rows: list[dict] = []

    def add(src, dst, amount, t, ttype, label):
        rows.append({
            "source_from_account": names[src], "source_to_account": names[dst],
            "amount": round(float(amount), 2), "timestamp": t, "transaction_type": ttype,
            "is_suspicious_label": int(label),
        })

    def rand_time():
        return BASE_TIME + pd.Timedelta(seconds=int(rng.integers(0, HORIZON_DAYS * 86400)))

    n_merchants, n_payroll = 30, 12
    merchants = np.arange(0, n_merchants)
    payroll = np.arange(n_merchants, n_merchants + n_payroll)
    ordinary = np.arange(n_merchants + n_payroll, n_accounts)

    # --- ordinary traffic -------------------------------------------------------------------------
    send_w = rng.lognormal(0.0, 1.0, n_accounts)
    recv_w = rng.lognormal(0.0, 1.0, n_accounts)
    recv_w[merchants] *= 25.0
    send_w[merchants] *= 0.2
    send_w /= send_w.sum()
    recv_w /= recv_w.sum()
    senders = rng.choice(n_accounts, size=n_normal_tx, p=send_w)
    receivers = rng.choice(n_accounts, size=n_normal_tx, p=recv_w)
    amounts = rng.lognormal(8.0, 1.2, n_normal_tx)
    for s, r, a in zip(senders, receivers, amounts):
        if s == r:
            continue
        add(s, r, a, rand_time(), "PAYMENT" if rng.random() < 0.6 else "TRANSFER", 0)

    # --- benign payroll bursts (fan-out, label 0) -------------------------------------------------
    for p in payroll:
        staff = rng.choice(ordinary, size=int(rng.integers(20, 41)), replace=False)
        salary = rng.lognormal(10.0, 0.4)
        for month in range(3):
            t0 = BASE_TIME + pd.Timedelta(days=28 * month + 1, hours=int(rng.integers(0, 6)))
            for e in staff:
                add(p, e, salary * rng.uniform(0.9, 1.1), t0 + pd.Timedelta(minutes=int(rng.integers(0, 90))),
                    "TRANSFER", 0)

    # --- benign rapid pass-through pairs (label 0): same shape as layering, but short ------------
    for _ in range(100):
        a, b, c = rng.choice(ordinary, size=3, replace=False)
        amt = rng.lognormal(9.0, 0.8)
        t = rand_time()
        add(a, b, amt, t, "TRANSFER", 0)
        add(b, c, amt * rng.uniform(0.9, 0.99), t + pd.Timedelta(minutes=int(rng.integers(5, 600))), "TRANSFER", 0)

    # --- laundering patterns (label 1) --------------------------------------------------------------
    kinds = ["cycle", "scatter_gather", "chain"]
    for i in range(n_patterns):
        kind = kinds[i % 3]
        amt = rng.lognormal(9.5, 1.0)
        t0 = rand_time()
        if kind == "cycle":
            k = int(rng.integers(3, 7))
            accts = rng.choice(ordinary, size=k, replace=False)
            for j in range(k):
                add(accts[j], accts[(j + 1) % k], amt * (0.98 ** j),
                    t0 + pd.Timedelta(minutes=int(rng.integers(5, 240)) * (j + 1)), "TRANSFER", 1)
        elif kind == "scatter_gather":
            m = int(rng.integers(3, 9))
            accts = rng.choice(ordinary, size=m + 2, replace=False)
            src, sink, mules = accts[0], accts[1], accts[2:]
            for j, mule in enumerate(mules):
                add(src, mule, amt / m, t0 + pd.Timedelta(minutes=int(rng.integers(1, 60))), "TRANSFER", 1)
                add(mule, sink, 0.95 * amt / m, t0 + pd.Timedelta(hours=int(rng.integers(1, 12)),
                                                                    minutes=int(rng.integers(0, 59))),
                    "TRANSFER", 1)
        else:
            k = int(rng.integers(4, 8))
            accts = rng.choice(ordinary, size=k, replace=False)
            for j in range(k - 1):
                add(accts[j], accts[j + 1], amt * (0.96 ** j),
                    t0 + pd.Timedelta(minutes=int(rng.integers(10, 120)) * (j + 1)), "TRANSFER", 1)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # no row-order leakage
    df["source_name"] = "fixture"
    df["source_tx_id"] = [f"FXT{i:07d}" for i in range(len(df))]
    df["currency"] = "INR"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df[CANONICAL_COLS]
