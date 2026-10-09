"""Test helpers: build the fixture dataset once per test run into a temp directory."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data" / "scripts"))

_ROOT = None


def built_dataset() -> Path:
    global _ROOT
    if _ROOT is None:
        _ROOT = Path(tempfile.mkdtemp(prefix="kycaml_test_"))
        os.environ["KYCAML_DATA_DIR"] = str(_ROOT)
        r = subprocess.run([sys.executable, str(REPO / "data" / "scripts" / "run_pipeline.py"), "--source", "fixture"],
                           env=dict(os.environ), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("pipeline failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
    return _ROOT


def fresh_db_copy() -> Path:
    root = built_dataset()
    dst = Path(tempfile.mkdtemp(prefix="kycaml_db_")) / "test.sqlite"
    shutil.copy(root / "db" / "kycaml.sqlite", dst)
    return dst
