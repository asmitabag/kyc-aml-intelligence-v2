"""Inventory the raw source datasets (owner: Saloni).

Licensed datasets (MIDV, Replay-Attack, OULU-NPU, IBM AML) require registration / licence acceptance, so this
script does NOT download them. Obtain each one from its official site, accept its licence, and place it under
data/raw/<source_name>/. Then run this script: it records file counts, sizes, SHA-256 hashes and the licence
status you must fill in, and writes data/manifests/source_inventory.json (used by the dataset manifest).

    python data/scripts/download_sources.py            # inventory everything found in data/raw
    python data/scripts/download_sources.py --no-hash  # skip hashing very large files

Raw data is never committed to Git (see .gitignore).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import MANIFESTS_DIR, RAW_DIR, ensure_dirs, sha256_file, utc_now_iso, write_json

KNOWN_SOURCES = {
    "midv": {"purpose": "identity-document detection / OCR / document research (KYC document data)",
             "labels_to_verify": "field annotations, document quads. Tampering labels are NOT guaranteed."},
    "replay_attack": {"purpose": "face liveness: real vs print/replay attack videos",
                      "labels_to_verify": "client (subject) IDs, real/attack, official train/devel/test split"},
    "oulu_npu": {"purpose": "face liveness: real vs print/replay attack videos",
                 "labels_to_verify": "user IDs, presentation-attack instrument, official protocols"},
    "ibm_aml": {"purpose": "AML transactions for GNN training",
                "labels_to_verify": "Is Laundering column, bank/account ID fields, currency names"},
    "amlsim": {"purpose": "AML transactions for GNN training (simulator output)",
               "labels_to_verify": "alert/SAR flag column, timestamp/step convention"},
}


def inventory_dir(path: Path, do_hash: bool) -> dict:
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    suffixes: dict[str, int] = {}
    for p in files:
        suffixes[p.suffix.lower()] = suffixes.get(p.suffix.lower(), 0) + 1
    entry = {
        "path": str(path), "file_count": len(files), "total_bytes": total, "extensions": suffixes,
        "sample_files": [str(p.relative_to(path)) for p in files[:5]],
    }
    if do_hash:
        entry["sha256"] = {str(p.relative_to(path)): sha256_file(p) for p in files[:2000]}
        entry["hash_note"] = "hashes recorded for the first 2000 files only" if len(files) > 2000 else "all files hashed"
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-hash", action="store_true")
    args = ap.parse_args(argv)
    ensure_dirs()
    report = {"generated_at": utc_now_iso(), "sources": {}}
    for name, meta in KNOWN_SOURCES.items():
        d = RAW_DIR / name
        if d.is_dir():
            entry = inventory_dir(d, not args.no_hash)
            entry["status"] = "PRESENT"
        else:
            entry = {"status": "MISSING", "expected_path": str(d)}
        entry.update({
            "purpose": meta["purpose"], "labels_to_verify": meta["labels_to_verify"],
            "version": "FILL_IN", "licence": "FILL_IN", "licence_allows_redistribution": "FILL_IN",
        })
        report["sources"][name] = entry
        print(f"{name:14s} {entry['status']}" + (f"  files={entry['file_count']}" if entry["status"] == "PRESENT" else ""))
    write_json(MANIFESTS_DIR / "source_inventory.json", report)
    print(f"\nWrote {MANIFESTS_DIR / 'source_inventory.json'}. Fill in version/licence fields by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
