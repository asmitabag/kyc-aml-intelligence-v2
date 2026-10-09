"""Write data/manifests/dataset_manifest.json for a release (owner: Saloni).

Run AFTER validate_dataset.py passes. The manifest is the contract between Saloni's dataset and everyone else's
models: sources, counts, ID-mapping method, data origins, label meaning, split rules, assumptions, file hashes.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import (DATASET_VERSION, DEFAULT_SEED, MANIFESTS_DIR, PROCESSED_DIR, SPLITS_DIR, ensure_dirs,
                    read_json, sha256_file, utc_now_iso, write_json)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=DATASET_VERSION)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    ensure_dirs()

    validation = read_json(MANIFESTS_DIR / "validation_report.json")
    if not validation["passed"]:
        raise SystemExit("Refusing to publish: validation_report.json has errors. Fix them first.")
    inventory_path = MANIFESTS_DIR / "source_inventory.json"
    inventory = read_json(inventory_path) if inventory_path.exists() else {"note": "no raw sources inventoried"}

    cust = pd.read_csv(PROCESSED_DIR / "customers.csv", dtype=str)
    tx = pd.read_csv(PROCESSED_DIR / "transactions.csv", dtype=str)
    kyc = pd.read_csv(PROCESSED_DIR / "kyc_results.csv", dtype=str)
    maps = pd.read_csv(PROCESSED_DIR / "id_mappings.csv", dtype=str)
    source_name = maps["source_name"].iloc[0]
    files = {str(p.relative_to(p.parents[1])): sha256_file(p)
             for p in sorted(list(PROCESSED_DIR.glob("*.csv")) + list(SPLITS_DIR.glob("*.csv")))}

    manifest = {
        "dataset_version": args.version,
        "built_at": utc_now_iso(),
        "seed": args.seed,
        "transaction_source": source_name,
        "raw_source_inventory": inventory,
        "counts": validation["statistics"]["rows"] | {"quarantined_transactions":
                                                      validation["statistics"]["quarantined_transactions"]},
        "data_origin_breakdown": {
            "customers": cust["data_origin"].value_counts().to_dict(),
            "transactions": tx["data_origin"].value_counts().to_dict(),
        },
        "labels": {
            "statistics": validation["statistics"]["labels"],
            "transaction_label": "source label: transaction is part of a known laundering/suspicious pattern",
            "account_label": "1 if the account sends or receives >=1 suspicious-labelled transaction",
            "inference_rule": "labels are training/evaluation targets only and are never inference features",
        },
        "canonical_ids": {
            "customer_id": "C######", "account_id": "A######", "transaction_id": "T######", "kyc_id": "KYC######",
            "method": "accounts numbered by a seeded hash of the source ID (no row-order or label information); "
                      "transactions numbered by time. Mapping in processed/id_mappings.csv.",
            "stability": "IDs are deterministic for the same source data + seed. A new release builds a new mapping.",
        },
        "identity_links": {
            "customer_to_account": "SYNTHETIC_ASSOCIATION: the AML sources have no customers; each customer is a "
                                   "synthetic person assigned 1-3 accounts by a seeded, label-blind process.",
            "kyc_to_aml": "no real person link exists between biometric/document sources and AML sources; none is claimed.",
        },
        "kyc_results": {
            "rows": int(len(kyc)),
            "origin": "SIMULATED by data/scripts/build_kyc_features.py (independent of AML labels, model_version "
                      "synthetic-kyc-sim-v1). Replace with Asmita's real outputs for genuine KYC-aware results.",
        },
        "splits": {
            "level": "account",
            "rule": "laundering groups (connected suspicious subgraphs) are kept whole inside one split; benign "
                    "accounts assigned by seeded hash; stats fitted on train only",
            "setting": "transductive graph (all transactions visible); validation/test labels never used",
            "sizes": validation["statistics"]["splits"],
        },
        "biometric_manifests": {
            "liveness_manifest_rows": validation["statistics"]["rows"].get("liveness_manifest", 0),
            "document_manifest_rows": validation["statistics"]["rows"].get("document_manifest", 0),
            "split_rule": "subject-disjoint; official dataset splits preserved where defined",
        },
        "assumptions_and_limitations": [
            "Fixture data (transaction_source=fixture) is fully synthetic; metrics on it prove the pipeline works, not research claims.",
            "IBM AML timestamps assumed UTC; AMLSim has no currency/real-time columns (defaults documented in prepare_transactions.py).",
            "MIDV provides no forgery labels: document tampering_label is UNAVAILABLE unless another labelled source is added.",
        ],
        "validation": {"passed": validation["passed"], "warnings": validation["warnings"]},
        "file_sha256": files,
    }
    write_json(MANIFESTS_DIR / "dataset_manifest.json", manifest)
    print(f"Published {args.version} manifest -> {MANIFESTS_DIR / 'dataset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
