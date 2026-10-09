"""Load the published dataset for training (owner: Saloni). Read-only: never modifies canonical files."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from services.aml import features as F


def data_dir() -> Path:
    return Path(os.environ.get("KYCAML_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


@dataclass
class AmlDataset:
    accounts: pd.DataFrame
    transactions: pd.DataFrame
    kyc_results: pd.DataFrame
    labels: pd.Series  # account_id -> 0/1 (targets only)
    split: pd.Series   # account_id -> train/validation/test
    manifest_version: str


def load_dataset(root: Path | None = None) -> AmlDataset:
    root = Path(root) if root else data_dir()
    proc, splits = root / "processed", root / "splits"
    accounts = pd.read_csv(proc / "accounts.csv", dtype=str)
    tx = pd.read_csv(proc / "transactions.csv", dtype=str)
    kyc = pd.read_csv(proc / "kyc_results.csv", dtype=str)
    lab = pd.read_csv(proc / "aml_labels.csv", dtype={"entity_id": str})
    labels = lab[lab["entity_type"] == "ACCOUNT"].set_index("entity_id")["label"].astype(int)
    parts = []
    for name in ("train", "validation", "test"):
        ids = pd.read_csv(splits / f"{name}_ids.csv", dtype=str)["entity_id"]
        parts.append(pd.Series(name, index=ids))
    split = pd.concat(parts)
    manifest = root / "manifests" / "dataset_manifest.json"
    version = "unknown"
    if manifest.exists():
        import json
        version = json.loads(manifest.read_text())["dataset_version"]
    return AmlDataset(accounts, tx, kyc, labels.reindex(accounts["account_id"]), split.reindex(accounts["account_id"]),
                      version)


def build_feature_frame(ds: AmlDataset, use_kyc: bool) -> pd.DataFrame:
    """Raw (unscaled) features for every account. Labels are not an input here."""
    feats = F.compute_account_features(ds.transactions, ds.accounts["account_id"])
    if use_kyc:
        feats = feats.join(F.kyc_features(ds.accounts, ds.kyc_results))
    return feats
