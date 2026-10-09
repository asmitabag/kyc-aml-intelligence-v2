"""Shared helpers for the dataset pipeline (owner: Saloni)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Generated data can be redirected (used by tests) without touching the repo's data/ folder.
DATA_DIR = Path(os.environ.get("KYCAML_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MANIFESTS_DIR = DATA_DIR / "manifests"
DB_DIR = DATA_DIR / "db"
SCHEMA_SQL = REPO_ROOT / "data" / "schema.sql"

DEFAULT_SEED = 20261009
DATASET_VERSION = "dataset_v1"

ENUMS = {
    "customer_type": {"INDIVIDUAL", "BUSINESS"},
    "customer_data_origin": {"SYNTHETIC", "CONSENTED_DEMO", "IMPORTED"},
    "transaction_data_origin": {"SYNTHETIC", "IMPORTED"},
    "kyc_component_status": {"PASS", "REVIEW", "FAIL", "NOT_RUN"},
    "kyc_overall_status": {"VERIFIED", "REVIEW", "REJECTED", "PENDING"},
}
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")

PROCESSED_TABLE_COLUMNS = {
    "customers": ["customer_id", "full_name", "date_of_birth", "created_at", "customer_type", "data_origin"],
    "accounts": ["account_id", "customer_id", "account_type", "currency", "created_at"],
    "kyc_results": [
        "kyc_id", "customer_id", "document_authenticity_score", "face_match_score", "liveness_score",
        "kyc_risk_score", "document_status", "face_match_status", "liveness_status", "overall_status",
        "model_version", "created_at",
    ],
    "documents": ["document_id", "customer_id", "document_type", "storage_uri", "data_origin", "created_at"],
    "transactions": [
        "transaction_id", "from_account_id", "to_account_id", "amount", "currency", "timestamp",
        "transaction_type", "data_origin", "is_suspicious_label",
    ],
    "id_mappings": [
        "source_name", "source_entity_type", "source_id", "canonical_entity_type", "canonical_id",
        "link_type", "notes",
    ],
}


def ensure_dirs() -> None:
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, SPLITS_DIR, MANIFESTS_DIR, DB_DIR):
        d.mkdir(parents=True, exist_ok=True)


def fmt_id(prefix: str, n: int, width: int = 6) -> str:
    return f"{prefix}{n:0{width}d}"


def stable_hash(text: str, seed: int = DEFAULT_SEED) -> int:
    """Deterministic 64-bit hash. Used to order/split entities independently of source row order."""
    return int(hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:16], 16)


def hash_split(key: str, seed: int = DEFAULT_SEED, fractions=(0.70, 0.15, 0.15)) -> str:
    u = stable_hash(key, seed) / float(2 ** 64)
    if u < fractions[0]:
        return "train"
    if u < fractions[0] + fractions[1]:
        return "validation"
    return "test"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_iso_z(series):
    """pandas Series of datetimes -> ISO 8601 UTC strings ending in Z."""
    import pandas as pd

    return pd.to_datetime(series, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
