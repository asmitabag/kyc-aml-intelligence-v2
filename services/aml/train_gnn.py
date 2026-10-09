"""Train the GNN account-risk model (owner: Saloni).

    python -m services.aml.train_gnn --run gnn_tx                 # transaction graph only
    python -m services.aml.train_gnn --run gnn_tx_kyc --use-kyc   # + KYC features (retrains: input space differs)

Protocol: weighted BCE on train accounts; early stopping on validation PR-AUC; decision threshold chosen on
validation and frozen; test metrics reported once at the end. Saves a checkpoint holding weights, architecture,
feature names, scaler, threshold and dataset version, so inference always matches training.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score

from services.aml import features as F
from services.aml.dataio import build_feature_frame, load_dataset
from services.aml.gnn import DirectedGraphSAGE, build_graph
from services.aml.metrics import best_f1_threshold, compute_metrics

ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = "gnn-v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(run_name: str, use_kyc: bool, epochs: int, hidden: int, layers: int, lr: float, dropout: float,
          patience: int, seed: int, root: Path | None = None, report_dir: Path | None = None,
          artifact_dir: Path | None = None) -> dict:
    set_seed(seed)
    ds = load_dataset(root)
    feats = build_feature_frame(ds, use_kyc)
    ids = ds.accounts["account_id"].tolist()
    split = ds.split.to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"
    y = ds.labels.to_numpy()

    scaler = F.fit_scaler(feats[tr])  # TRAIN statistics only
    x = F.apply_scaler(feats, scaler)
    data = build_graph(x, ids, ds.transactions, y)
    tr_m, va_m, te_m = (torch.from_numpy(m) for m in (tr, va, te))

    model = DirectedGraphSAGE(x.shape[1], hidden, layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = max(float(y[tr].sum()), 1.0)
    pos_weight = torch.tensor((tr.sum() - pos) / pos, dtype=torch.float32)

    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = Fn.binary_cross_entropy_with_logits(logits[tr_m], data.y[tr_m], pos_weight=pos_weight)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(data.x, data.edge_index)).numpy()
        ap = average_precision_score(y[va], prob[va]) if len(np.unique(y[va])) == 2 else 0.0
        history.append({"epoch": epoch, "train_loss": float(loss), "val_pr_auc": float(ap)})
        if ap > best_ap:
            best_ap, best_epoch, bad = ap, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(data.x, data.edge_index)).numpy()
    thr = best_f1_threshold(y[va], scores[va])  # chosen on validation, then frozen
    result = {
        "run": run_name, "model": "DirectedGraphSAGE", "model_version": MODEL_VERSION, "use_kyc": use_kyc,
        "seed": seed, "dataset_version": ds.manifest_version, "feature_names": list(feats.columns),
        "hyperparameters": {"hidden": hidden, "layers": layers, "lr": lr, "dropout": dropout, "epochs_run": len(history)},
        "best_epoch": best_epoch, "validation": compute_metrics(y[va], scores[va], thr),
        "test": compute_metrics(y[te], scores[te], thr),
    }

    rep = (report_dir or ROOT / "reports") / run_name
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "metrics.json").write_text(json.dumps(result, indent=2))
    (rep / "history.json").write_text(json.dumps(history))
    pd.DataFrame({"account_id": ids, "split": split, "label": y, "score": scores}).to_csv(rep / "predictions.csv", index=False)

    art = (artifact_dir or ROOT / "artifacts") / run_name
    art.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "architecture": {"in_dim": x.shape[1], "hidden": hidden, "layers": layers, "dropout": dropout},
        "feature_names": list(feats.columns), "scaler": scaler, "threshold": thr, "use_kyc": use_kyc,
        "model_version": MODEL_VERSION + ("-kyc" if use_kyc else ""), "dataset_version": ds.manifest_version, "seed": seed,
    }, art / "checkpoint.pt")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="gnn_tx")
    ap.add_argument("--use-kyc", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--data-root")
    ap.add_argument("--report-dir")
    ap.add_argument("--artifact-dir")
    a = ap.parse_args(argv)
    r = train(a.run, a.use_kyc, a.epochs, a.hidden, a.layers, a.lr, a.dropout, a.patience, a.seed,
              Path(a.data_root) if a.data_root else None, Path(a.report_dir) if a.report_dir else None,
              Path(a.artifact_dir) if a.artifact_dir else None)
    t = r["test"]
    print(f"{a.run}: best_epoch={r['best_epoch']} test P={t['precision']:.3f} R={t['recall']:.3f} "
          f"F1={t['f1']:.3f} PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
