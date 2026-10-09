"""Generate SIMULATED KYC results for synthetic customers (owner: Saloni).

Why this exists: the KYC-aware GNN experiment needs KYC attributes attached to AML accounts, but the biometric
and AML source datasets contain different (unrelated) people. Rather than fake a correspondence, we generate
KYC attributes by an INDEPENDENT, documented random process.

HARD RULE (research constraint): this script reads customers.csv ONLY. It never reads aml_labels.csv or the
transaction labels, so the simulated scores cannot be derived from fraud labels. validate_dataset.py additionally
measures the association between these scores and the labels and warns if it is not ~0.

Consequence to report honestly: because the simulated KYC risk is independent of the labels by construction,
a KYC-aware GNN is NOT expected to beat the transaction-only GNN on this data. The experiment only becomes
informative with genuine KYC outputs from Asmita's models on genuinely linked records.

Simulation assumptions (documented in dataset_manifest.json)
  * 90% of customers have a KYC attempt; 10% have none (missing, not zero risk).
  * a latent "poor evidence" flag with probability 0.08 draws lower component scores.
  * kyc_risk_score = 1 - (0.40*document + 0.30*face + 0.30*liveness)   [placeholder, NOT Asmita's S_kyc formula]
  * statuses: score >= 0.80 PASS, 0.50-0.80 REVIEW, < 0.50 FAIL.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import DEFAULT_SEED, PROCESSED_DIR, PROCESSED_TABLE_COLUMNS, fmt_id

MODEL_VERSION = "synthetic-kyc-sim-v1"


def status(score: float) -> str:
    return "PASS" if score >= 0.80 else ("REVIEW" if score >= 0.50 else "FAIL")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--coverage", type=float, default=0.90)
    args = ap.parse_args(argv)

    customers = pd.read_csv(PROCESSED_DIR / "customers.csv", dtype=str)  # <- the ONLY input
    rng = np.random.default_rng(args.seed + 7)
    rows = []
    n = 0
    for cid, created in zip(customers["customer_id"], customers["created_at"]):
        if rng.random() > args.coverage:
            continue
        poor = rng.random() < 0.08
        a, b = (4, 3) if poor else (14, 2)  # Beta(a, b): mean 0.57 vs 0.88
        doc, face, live = (float(np.clip(rng.beta(a, b), 0, 1)) for _ in range(3))
        risk = 1.0 - (0.40 * doc + 0.30 * face + 0.30 * live)
        st = [status(doc), status(face), status(live)]
        overall = "REJECTED" if "FAIL" in st else ("REVIEW" if "REVIEW" in st else "VERIFIED")
        n += 1
        rows.append({
            "kyc_id": fmt_id("KYC", n), "customer_id": cid,
            "document_authenticity_score": round(doc, 4), "face_match_score": round(face, 4),
            "liveness_score": round(live, 4), "kyc_risk_score": round(risk, 4),
            "document_status": st[0], "face_match_status": st[1], "liveness_status": st[2],
            "overall_status": overall, "model_version": MODEL_VERSION, "created_at": created,
        })
    out = pd.DataFrame(rows, columns=PROCESSED_TABLE_COLUMNS["kyc_results"])
    out.to_csv(PROCESSED_DIR / "kyc_results.csv", index=False)
    print(f"simulated KYC rows: {len(out):,} of {len(customers):,} customers (model_version={MODEL_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
