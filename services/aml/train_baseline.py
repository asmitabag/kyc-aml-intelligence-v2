"""Transaction-feature baseline (no graph message passing) for comparison with the GNN (owner: Saloni).

    python -m services.aml.train_baseline --run baseline_tx
    python -m services.aml.train_baseline --run baseline_tx_kyc --use-kyc

Uses scikit-learn's HistGradientBoostingClassifier on the SAME account features as the GNN. It is the "simpler
transaction-only model" the GNN must beat to justify graph learning. (Sounak's XGBoost is a separate deliverable.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from services.aml import features as F
from services.aml.dataio import build_feature_frame, load_dataset
from services.aml.metrics import best_f1_threshold, compute_metrics

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def run(run_name: str, use_kyc: bool, seed: int, root: Path | None = None, out_dir: Path | None = None) -> dict:
    ds = load_dataset(root)
    feats = build_feature_frame(ds, use_kyc)
    ids = ds.accounts["account_id"]
    tr, va, te = (ds.split == s for s in ("train", "validation", "test"))
    tr, va, te = tr.to_numpy(), va.to_numpy(), te.to_numpy()
    y = ds.labels.to_numpy()
    scaler = F.fit_scaler(feats[tr])  # train-only statistics (tree models don't need it; kept for parity)
    X = F.apply_scaler(feats, scaler)

    pos = max(int(y[tr].sum()), 1)
    w = np.where(y[tr] == 1, (tr.sum() - pos) / pos, 1.0)
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=seed)
    model.fit(X[tr], y[tr], sample_weight=w)
    scores = model.predict_proba(X)[:, 1]
    thr = best_f1_threshold(y[va], scores[va])
    result = {
        "run": run_name, "model": "HistGradientBoostingClassifier", "use_kyc": use_kyc, "seed": seed,
        "dataset_version": ds.manifest_version, "feature_names": list(feats.columns),
        "validation": compute_metrics(y[va], scores[va], thr), "test": compute_metrics(y[te], scores[te], thr),
    }
    out = (out_dir or REPORTS) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({"account_id": ids, "split": ds.split.to_numpy(), "label": y, "score": scores}
                 ).to_csv(out / "predictions.csv", index=False)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="baseline_tx")
    ap.add_argument("--use-kyc", action="store_true")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--data-root")
    ap.add_argument("--out-dir")
    a = ap.parse_args(argv)
    r = run(a.run, a.use_kyc, a.seed, Path(a.data_root) if a.data_root else None, Path(a.out_dir) if a.out_dir else None)
    t = r["test"]
    print(f"{a.run}: test P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} "
          f"PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
