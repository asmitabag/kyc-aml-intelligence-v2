"""Create the SQLite application database from the published dataset (owner: Saloni).

    python data/scripts/build_database.py [--out data/db/kycaml.sqlite]

Loads customers, accounts, kyc_results, documents, transactions and id_mappings with foreign keys enforced.
risk_assessments / alerts / sar_drafts are created empty (Sounak's orchestration fills them).
"""
from __future__ import annotations

import argparse
import sqlite3

import pandas as pd

from common import DB_DIR, PROCESSED_DIR, PROCESSED_TABLE_COLUMNS, SCHEMA_SQL, ensure_dirs

LOAD_ORDER = ["customers", "accounts", "kyc_results", "documents", "transactions", "id_mappings"]
REAL_COLS = {"amount", "document_authenticity_score", "face_match_score", "liveness_score", "kyc_risk_score"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DB_DIR / "kycaml.sqlite"))
    args = ap.parse_args(argv)
    ensure_dirs()
    from pathlib import Path
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(out)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
    for name in LOAD_ORDER:
        cols = PROCESSED_TABLE_COLUMNS[name]
        df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype=str, keep_default_na=False)
        records = []
        for row in df[cols].itertuples(index=False):
            rec = []
            for col, val in zip(cols, row):
                if val == "":
                    rec.append(None)  # missing stays NULL, never 0
                elif col in REAL_COLS:
                    rec.append(float(val))
                elif col == "is_suspicious_label":
                    rec.append(int(float(val)))
                else:
                    rec.append(val)
            records.append(tuple(rec))
        conn.executemany(f"INSERT INTO {name} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", records)
        print(f"loaded {name:12s} {len(records):8,d} rows")
    conn.commit()
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    conn.close()
    if violations:
        raise SystemExit(f"foreign key violations: {violations[:5]}")
    print(f"database ready: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
