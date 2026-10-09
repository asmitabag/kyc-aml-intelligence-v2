"""Evaluation metrics shared by the baseline and the GNN (owner: Saloni)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def best_f1_threshold(y_true, scores) -> float:
    """Threshold maximising F1. Choose it on VALIDATION data only, then freeze it for the test set."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 197)))
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        f = f1_score(y_true, scores >= t, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def compute_metrics(y_true, scores, threshold: float) -> dict:
    y_true, scores = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)) if two_classes else None,
        "roc_auc": float(roc_auc_score(y_true, scores)) if two_classes else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"n": int(len(y_true)), "positives": int(y_true.sum()), "prevalence": float(y_true.mean())},
    }
