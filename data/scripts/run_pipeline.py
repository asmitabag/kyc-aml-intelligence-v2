"""Run the whole dataset pipeline end to end (owner: Saloni).

    python data/scripts/run_pipeline.py --source fixture
    python data/scripts/run_pipeline.py --source ibm_aml --input data/raw/ibm_aml/HI-Small_Trans.csv
    KYCAML_DATA_DIR=/tmp/x python data/scripts/run_pipeline.py --source fixture    # redirect output

Any extra arguments after `--` are passed to prepare_kyc_data.py (e.g. -- --oulu data/raw/oulu_npu).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}")
    r = subprocess.run([sys.executable, str(HERE / script), *args])
    if r.returncode != 0:
        raise SystemExit(f"step failed: {script} (exit {r.returncode})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["fixture", "ibm_aml", "amlsim"], default="fixture")
    ap.add_argument("--input")
    ap.add_argument("--version", default="dataset_v1")
    ap.add_argument("kyc_args", nargs="*", help="extra args for prepare_kyc_data.py")
    args = ap.parse_args(argv)

    prep = ["--source", args.source] + (["--input", args.input] if args.input else [])
    run("download_sources.py", "--no-hash")
    run("prepare_transactions.py", *prep)
    run("build_registry.py")
    run("prepare_kyc_data.py", *args.kyc_args)
    run("build_kyc_features.py")
    run("build_splits.py")
    run("validate_dataset.py")
    run("publish_dataset.py", "--version", args.version)
    run("build_database.py")
    print("\nPipeline finished. Manifest: data/manifests/dataset_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
