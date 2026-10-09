"""Compare runs and analyse errors (owner: Saloni). Needs no torch: it reads the saved predictions.

    python -m services.aml.evaluate --runs baseline_tx gnn_tx gnn_tx_kyc

Writes reports/aml_evaluation.md and reports/aml_evaluation.json containing, per run, the TEST metrics
(precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix) at the validation-selected threshold, plus a false-negative
and false-positive analysis. Compare: (1) transaction-feature baseline, (2) GNN on the transaction graph,
(3) GNN + KYC features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from services.aml import features as F
from services.aml.dataio import load_dataset

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def error_analysis(pred: pd.DataFrame, threshold: float, ds, top: int = 10) -> dict:
    test = pred[pred["split"] == "test"].copy()
    test["pred"] = (test["score"] >= threshold).astype(int)
    feats = F.compute_account_features(ds.transactions, test["account_id"]).reset_index()
    test = test.merge(feats, on="account_id")
    fn = test[(test["label"] == 1) & (test["pred"] == 0)].sort_values("score")
    fp = test[(test["label"] == 0) & (test["pred"] == 1)].sort_values("score", ascending=False)
    cols = ["account_id", "score", "out_count", "in_count", "flow_through_ratio", "rapid_out_after_in_frac"]
    return {
        "false_negatives": len(fn), "false_positives": len(fp),
        "missed_positives_mean_features": fn[F.TX_FEATURE_NAMES].mean().round(3).to_dict() if len(fn) else {},
        "caught_positives_mean_features": test[(test["label"] == 1) & (test["pred"] == 1)][F.TX_FEATURE_NAMES].mean().round(3).to_dict(),
        "lowest_scored_missed_accounts": fn[cols].head(top).round(4).to_dict("records"),
        "highest_scored_false_alarms": fp[cols].head(top).round(4).to_dict("records"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["baseline_tx", "gnn_tx", "gnn_tx_kyc"])
    ap.add_argument("--report-dir")
    ap.add_argument("--data-root")
    a = ap.parse_args(argv)
    rep = Path(a.report_dir) if a.report_dir else REPORTS
    ds = load_dataset(Path(a.data_root) if a.data_root else None)

    results, lines = {}, []
    for run in a.runs:
        mpath = rep / run / "metrics.json"
        if not mpath.exists():
            print(f"skip {run}: no metrics at {mpath}")
            continue
        m = json.loads(mpath.read_text())
        pred = pd.read_csv(rep / run / "predictions.csv", dtype={"account_id": str})
        m["error_analysis"] = error_analysis(pred, m["test"]["threshold"], ds)
        results[run] = m

    if not results:
        raise SystemExit("no runs found")
    any_run = next(iter(results.values()))
    lines += ["# AML evaluation report", "",
              f"Dataset: `{any_run['dataset_version']}`. Metrics are on the held-out TEST accounts at the threshold "
              "chosen on validation. Test prevalence is the PR-AUC reference level (a random scorer's PR-AUC).", "",
              "| Run | Model | KYC | Precision | Recall | F1 | PR-AUC | ROC-AUC | TP | FP | FN | TN |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, m in results.items():
        t, c = m["test"], m["test"]["confusion_matrix"]
        lines.append(f"| {run} | {m['model']} | {'yes' if m['use_kyc'] else 'no'} | {t['precision']:.3f} | "
                     f"{t['recall']:.3f} | {t['f1']:.3f} | {t['pr_auc']:.3f} | {t['roc_auc']:.3f} | "
                     f"{c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |")
    prev = any_run["test"]["support"]
    lines += ["", f"Test accounts: {prev['n']}, positives: {prev['positives']} (prevalence {prev['prevalence']:.3f}).", "",
              "## Error analysis", ""]
    for run, m in results.items():
        e = m["error_analysis"]
        lines += [f"### {run}", f"- false negatives: {e['false_negatives']}, false positives: {e['false_positives']}",
                  f"- mean features of MISSED positives: `{e['missed_positives_mean_features']}`",
                  f"- mean features of CAUGHT positives: `{e['caught_positives_mean_features']}`", ""]
    lines += ["## How to read this", "",
              "- A GNN only 'wins' if its PR-AUC/F1 beats `baseline_tx` on the same test accounts.",
              "- `gnn_tx_kyc` vs `gnn_tx` tests whether KYC features add signal. With SIMULATED KYC that is independent of "
              "the labels, no gain is expected; do not claim one. A real test needs genuine KYC outputs on linked records.",
              "- Fixture data is synthetic: these numbers verify the pipeline, not the research hypothesis."]
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "aml_evaluation.md").write_text("\n".join(lines) + "\n")
    (rep / "aml_evaluation.json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines[:12 + len(results)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
