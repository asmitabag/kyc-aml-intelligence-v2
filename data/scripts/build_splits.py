"""Build leakage-aware train/validation/test splits for the AML accounts (owner: Saloni).

Reads  data/processed/{accounts,transactions,aml_labels}.csv
Writes data/splits/{train,validation,test}_ids.csv  (entity_type, entity_id, group_id)

Leakage prevention
  * Suspicious accounts that are connected through suspicious transactions form ONE group (a laundering ring /
    chain). A whole group goes to a single split, so a pattern is never half in train and half in test.
  * Positive groups are shuffled and spread across the three splits so each split has positives.
  * Remaining (benign) accounts are assigned by a seeded hash of their account_id.
  * Feature normalisation statistics are fitted on the train split only (enforced in services/aml/features.py).

Setting (documented): the graph is TRANSDUCTIVE. All transactions are visible when computing graph features, but
the labels of validation/test accounts are never used for training or for feature construction.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import DEFAULT_SEED, PROCESSED_DIR, SPLITS_DIR, ensure_dirs, hash_split


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    ensure_dirs()

    accounts = pd.read_csv(PROCESSED_DIR / "accounts.csv", dtype=str)
    tx = pd.read_csv(PROCESSED_DIR / "transactions.csv", dtype=str)
    labels = pd.read_csv(PROCESSED_DIR / "aml_labels.csv", dtype={"entity_id": str})
    acct_label = labels[labels["entity_type"] == "ACCOUNT"].set_index("entity_id")["label"]

    ids = accounts["account_id"].tolist()
    index = {a: i for i, a in enumerate(ids)}
    sus = tx[tx["is_suspicious_label"].astype(int) == 1]
    r = sus["from_account_id"].map(index).to_numpy()
    c = sus["to_account_id"].map(index).to_numpy()
    n = len(ids)
    adj = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, comp = connected_components(adj, directed=False)

    group = {}
    for a, i in index.items():
        group[a] = f"G{comp[i]:06d}" if acct_label.get(a, 0) == 1 else f"S{a}"

    pos_groups = sorted({g for a, g in group.items() if acct_label.get(a, 0) == 1})
    rng = np.random.default_rng(args.seed + 11)
    rng.shuffle(pos_groups)
    k = len(pos_groups)
    n_tr, n_va = int(round(0.70 * k)), int(round(0.15 * k))
    if k >= 3:
        n_va = max(n_va, 1)
        n_tr = min(n_tr, k - n_va - 1)
    group_split = {g: ("train" if i < n_tr else ("validation" if i < n_tr + n_va else "test"))
                   for i, g in enumerate(pos_groups)}

    split_of = {}
    for a in ids:
        g = group[a]
        split_of[a] = group_split[g] if g in group_split else hash_split("account:" + a, args.seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        members = [a for a in ids if split_of[a] == name]
        pd.DataFrame({"entity_type": "ACCOUNT", "entity_id": members, "group_id": [group[a] for a in members]}
                     ).to_csv(SPLITS_DIR / f"{name}_ids.csv", index=False)
        pos = int(sum(acct_label.get(a, 0) for a in members))
        print(f"{name:10s} accounts={len(members):6,d} positives={pos:5,d} ({pos / max(len(members), 1):.2%})")
    if k < 3:
        print("WARNING: fewer than 3 positive groups; at least one split has no positives. Use more data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
