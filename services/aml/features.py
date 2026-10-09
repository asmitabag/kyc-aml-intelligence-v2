"""Account-level features computed from transactions (owner: Saloni).

Used IDENTICALLY at training and at inference. The functions here never receive labels, so a label can not leak
into a feature. Normalisation statistics are fitted on the TRAIN split only and stored with the model.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

TX_FEATURE_NAMES = [
    "out_count", "in_count", "out_sum_log", "in_sum_log", "out_mean_log", "in_mean_log",
    "out_max_log", "in_max_log", "out_unique_cp", "in_unique_cp", "flow_through_ratio", "net_flow_ratio",
    "active_span_days", "tx_per_day", "reciprocal_frac", "self_loop_count", "rapid_out_after_in_frac",
]
KYC_FEATURE_NAMES = ["kyc_available", "kyc_risk_score"]


def prepare_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns features may use. is_suspicious_label is dropped here on purpose."""
    cols = ["transaction_id", "from_account_id", "to_account_id", "amount", "timestamp"]
    out = tx[cols].copy()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out


def compute_account_features(tx: pd.DataFrame, account_ids: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame indexed by account_id with TX_FEATURE_NAMES (zeros for accounts without transactions)."""
    tx = prepare_transactions(tx)
    idx = pd.Index(list(account_ids), name="account_id")
    out = pd.DataFrame(index=idx)

    g_out = tx.groupby("from_account_id")["amount"]
    g_in = tx.groupby("to_account_id")["amount"]
    out["out_count"] = g_out.size()
    out["in_count"] = g_in.size()
    out["out_sum"], out["in_sum"] = g_out.sum(), g_in.sum()
    out["out_mean"], out["in_mean"] = g_out.mean(), g_in.mean()
    out["out_max"], out["in_max"] = g_out.max(), g_in.max()
    out["out_unique_cp"] = tx.groupby("from_account_id")["to_account_id"].nunique()
    out["in_unique_cp"] = tx.groupby("to_account_id")["from_account_id"].nunique()

    # Reciprocity: share of an account's distinct counterparties it both sends to and receives from.
    pairs = tx[["from_account_id", "to_account_id"]].drop_duplicates()
    rev = pairs.rename(columns={"from_account_id": "to_account_id", "to_account_id": "from_account_id"})
    both = pairs.merge(rev, on=["from_account_id", "to_account_id"])
    recip = both.groupby("from_account_id").size()
    out["_recip"] = recip

    loops = tx[tx["from_account_id"] == tx["to_account_id"]].groupby("from_account_id").size()
    out["self_loop_count"] = loops

    # Active span from first to last transaction on either side.
    long = pd.concat([
        tx[["from_account_id", "timestamp"]].rename(columns={"from_account_id": "account_id"}),
        tx[["to_account_id", "timestamp"]].rename(columns={"to_account_id": "account_id"}),
    ])
    span = long.groupby("account_id")["timestamp"].agg(["min", "max"])
    out["active_span_days"] = (span["max"] - span["min"]).dt.total_seconds() / 86400.0

    # Outgoing transactions that happen within 24h after some incoming transaction at the same account.
    outs = tx[["from_account_id", "timestamp"]].rename(columns={"from_account_id": "account_id", "timestamp": "t"})
    ins = tx[["to_account_id", "timestamp"]].rename(columns={"to_account_id": "account_id", "timestamp": "t"})
    ins = ins.assign(t_in=ins["t"])
    merged = pd.merge_asof(outs.sort_values("t"), ins.sort_values("t"), on="t", by="account_id",
                           direction="backward", tolerance=pd.Timedelta("24h"), allow_exact_matches=False)
    rapid = merged.assign(hit=merged["t_in"].notna()).groupby("account_id")["hit"].sum()
    out["_rapid"] = rapid

    out = out.fillna(0.0)
    feats = pd.DataFrame(index=idx)
    for c in ("out_count", "in_count", "out_unique_cp", "in_unique_cp", "self_loop_count", "active_span_days"):
        feats[c] = out[c]
    for c in ("out_sum", "in_sum", "out_mean", "in_mean", "out_max", "in_max"):
        feats[c + "_log"] = np.log1p(out[c])
    hi, lo = np.maximum(out["in_sum"], out["out_sum"]), np.minimum(out["in_sum"], out["out_sum"])
    feats["flow_through_ratio"] = np.where(hi > 0, lo / np.where(hi > 0, hi, 1.0), 0.0)
    feats["net_flow_ratio"] = (out["in_sum"] - out["out_sum"]) / (out["in_sum"] + out["out_sum"] + 1.0)
    total = out["in_count"] + out["out_count"]
    feats["tx_per_day"] = total / (out["active_span_days"] + 1.0)
    feats["reciprocal_frac"] = out["_recip"] / np.maximum(out["out_unique_cp"], 1.0)
    feats["rapid_out_after_in_frac"] = out["_rapid"] / np.maximum(out["out_count"], 1.0)
    return feats[TX_FEATURE_NAMES].astype("float64")


def kyc_features(accounts: pd.DataFrame, kyc: pd.DataFrame | None) -> pd.DataFrame:
    """Per-account KYC features from the LATEST KYC result of the owning customer.

    Missing KYC is encoded as kyc_available=0 with kyc_risk_score=0 (the indicator carries the 'unknown' meaning,
    the 0 is a neutral fill, not a claim of zero risk).
    """
    idx = pd.Index(accounts["account_id"], name="account_id")
    out = pd.DataFrame(index=idx, data={"kyc_available": 0.0, "kyc_risk_score": 0.0})
    if kyc is None or len(kyc) == 0:
        return out
    k = kyc.copy()
    k["kyc_risk_score"] = pd.to_numeric(k["kyc_risk_score"], errors="coerce")
    k = k.dropna(subset=["kyc_risk_score"]).sort_values("created_at").groupby("customer_id").tail(1)
    per_cust = k.set_index("customer_id")["kyc_risk_score"]
    scores = accounts["customer_id"].map(per_cust)
    out["kyc_available"] = scores.notna().astype(float).to_numpy()
    out["kyc_risk_score"] = scores.fillna(0.0).to_numpy()
    return out


def fit_scaler(train_features: pd.DataFrame) -> dict:
    """Mean/std from TRAIN accounts only."""
    mean = train_features.mean()
    std = train_features.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return {"feature_names": list(train_features.columns), "mean": mean.tolist(), "std": std.tolist()}


def apply_scaler(features: pd.DataFrame, scaler: dict) -> np.ndarray:
    cols = scaler["feature_names"]
    missing = [c for c in cols if c not in features.columns]
    if missing:
        raise ValueError(f"features missing for this model: {missing}")
    x = (features[cols].to_numpy(dtype="float64") - np.asarray(scaler["mean"])) / np.asarray(scaler["std"])
    return x.astype("float32")
