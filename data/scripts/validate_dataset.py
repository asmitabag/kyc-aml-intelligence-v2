"""Validate a dataset release (owner: Saloni). Exit code 1 if any ERROR-level check fails.

Writes data/manifests/validation_report.json and validation_report.md
Checks: primary keys, foreign keys, enums, amounts, timestamps, currencies, label coverage, split integrity and
disjointness, biometric subject-disjointness, KYC ranges, and independence of simulated KYC from AML labels.
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

from common import (CURRENCY_RE, ENUMS, MANIFESTS_DIR, PROCESSED_DIR, PROCESSED_TABLE_COLUMNS, SPLITS_DIR,
                    ensure_dirs, utc_now_iso, write_json)

TS_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


class Report:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, detail="", severity="error"):
        self.checks.append({"name": name, "passed": bool(ok), "severity": severity, "detail": str(detail)})

    @property
    def errors(self):
        return [c for c in self.checks if not c["passed"] and c["severity"] == "error"]

    @property
    def warnings(self):
        return [c for c in self.checks if not c["passed"] and c["severity"] == "warning"]


def read(name, **kw):
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype=str, keep_default_na=False, **kw)


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    ensure_dirs()
    R = Report()
    t = {n: read(n) for n in ("customers", "accounts", "transactions", "kyc_results", "documents", "id_mappings")}
    labels = pd.read_csv(PROCESSED_DIR / "aml_labels.csv", dtype={"entity_id": str})
    stats = {"rows": {n: int(len(df)) for n, df in t.items()}}

    # columns / primary keys
    pks = {"customers": "customer_id", "accounts": "account_id", "transactions": "transaction_id",
           "kyc_results": "kyc_id", "documents": "document_id"}
    for name, df in t.items():
        R.check(f"{name}: columns match contract", list(df.columns) == PROCESSED_TABLE_COLUMNS[name], list(df.columns))
    for name, pk in pks.items():
        R.check(f"{name}: primary key unique", not t[name][pk].duplicated().any())
        R.check(f"{name}: primary key non-empty", (t[name][pk] != "").all())

    # foreign keys
    cust_ids, acct_ids = set(t["customers"]["customer_id"]), set(t["accounts"]["account_id"])
    R.check("accounts.customer_id -> customers", t["accounts"]["customer_id"].isin(cust_ids).all())
    R.check("transactions.from_account_id -> accounts", t["transactions"]["from_account_id"].isin(acct_ids).all(),
            "unresolved references must be quarantined")
    R.check("transactions.to_account_id -> accounts", t["transactions"]["to_account_id"].isin(acct_ids).all())
    R.check("kyc_results.customer_id -> customers", t["kyc_results"]["customer_id"].isin(cust_ids).all())
    R.check("documents.customer_id -> customers", t["documents"]["customer_id"].isin(cust_ids).all())
    R.check("every customer has >=1 account", cust_ids <= set(t["accounts"]["customer_id"]), severity="warning")

    # value rules
    tx = t["transactions"]
    amt = pd.to_numeric(tx["amount"], errors="coerce")
    R.check("transactions.amount > 0 and numeric", (amt > 0).all())
    R.check("transactions.timestamp is ISO 8601 UTC (Z)", tx["timestamp"].str.match(TS_RE).all())
    R.check("transactions.currency is 3 uppercase letters", tx["currency"].map(lambda c: bool(CURRENCY_RE.match(c))).all())
    R.check("transactions.data_origin enum", tx["data_origin"].isin(ENUMS["transaction_data_origin"]).all())
    R.check("customers enums", t["customers"]["customer_type"].isin(ENUMS["customer_type"]).all() and
            t["customers"]["data_origin"].isin(ENUMS["customer_data_origin"]).all())
    R.check("accounts.created_at ISO UTC", t["accounts"]["created_at"].str.match(TS_RE).all())
    self_loops = int((tx["from_account_id"] == tx["to_account_id"]).sum())
    R.check("self-loop transactions", self_loops == 0, f"{self_loops} self-loops (kept, counted as a feature)", "warning")
    dup = int(tx.duplicated(["from_account_id", "to_account_id", "amount", "timestamp"]).sum())
    R.check("no identical duplicate transactions", dup == 0, f"{dup} duplicates", "warning")

    # labels
    tl = labels[labels["entity_type"] == "TRANSACTION"]
    al = labels[labels["entity_type"] == "ACCOUNT"]
    R.check("aml_labels binary", labels["label"].isin([0, 1]).all())
    R.check("every transaction labelled", set(tl["entity_id"]) == set(tx["transaction_id"]))
    R.check("every account labelled", set(al["entity_id"]) == acct_ids)
    R.check("transactions.is_suspicious_label matches aml_labels",
            (tx.set_index("transaction_id")["is_suspicious_label"].astype(int)
             == tl.set_index("entity_id")["label"].reindex(tx["transaction_id"]).to_numpy()).all())
    stats["labels"] = {"transactions_positive": int(tl["label"].sum()), "transactions_total": int(len(tl)),
                       "accounts_positive": int(al["label"].sum()), "accounts_total": int(len(al)),
                       "account_prevalence": float(al["label"].mean()) if len(al) else None}

    # splits
    split_sets, split_groups = {}, {}
    for s in ("train", "validation", "test"):
        df = pd.read_csv(SPLITS_DIR / f"{s}_ids.csv", dtype=str)
        split_sets[s] = set(df["entity_id"])
        split_groups[s] = set(df["group_id"])
    union = set().union(*split_sets.values())
    R.check("splits are pairwise disjoint", sum(len(v) for v in split_sets.values()) == len(union))
    R.check("splits cover every account exactly once", union == acct_ids)
    R.check("no laundering group spans two splits",
            len(split_groups["train"] & split_groups["validation"]) == 0 and
            len(split_groups["train"] & split_groups["test"]) == 0 and
            len(split_groups["validation"] & split_groups["test"]) == 0)
    pos_by_split = {s: int(al[al["entity_id"].isin(v)]["label"].sum()) for s, v in split_sets.items()}
    R.check("each split contains positives", all(v > 0 for v in pos_by_split.values()), pos_by_split)
    stats["splits"] = {s: {"accounts": len(v), "positives": pos_by_split[s]} for s, v in split_sets.items()}

    # biometric manifests: subject-disjoint
    for name, subj_col in (("liveness_manifest", "subject_id"), ("document_manifest", "source_subject_id")):
        m = pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype=str, keep_default_na=False)
        stats["rows"][name] = int(len(m))
        if len(m):
            per_subject = m.groupby(subj_col)["split"].nunique()
            R.check(f"{name}: subject-disjoint splits", (per_subject <= 1).all(),
                    f"{int((per_subject > 1).sum())} subjects span >1 split")
        else:
            R.check(f"{name}: has rows", False, "empty (no KYC source supplied yet)", "warning")
    lm = pd.read_csv(PROCESSED_DIR / "liveness_manifest.csv", dtype=str, keep_default_na=False)
    if len(lm):
        R.check("liveness labels are REAL/SPOOF", lm["label"].isin(["REAL", "SPOOF"]).all())

    # KYC ranges + independence from labels
    k = t["kyc_results"]
    if len(k):
        for col in ("document_authenticity_score", "face_match_score", "liveness_score", "kyc_risk_score"):
            v = pd.to_numeric(k[col].replace("", pd.NA), errors="coerce").dropna()
            R.check(f"kyc_results.{col} in [0,1]", v.between(0, 1).all())
        R.check("kyc status enums", k["overall_status"].isin(ENUMS["kyc_overall_status"]).all() and
                k["document_status"].isin(ENUMS["kyc_component_status"]).all())
        if (k["model_version"].str.startswith("synthetic")).all():
            cust_label = (t["accounts"].merge(al.rename(columns={"entity_id": "account_id"})[["account_id", "label"]],
                                               on="account_id").groupby("customer_id")["label"].max())
            j = k.assign(risk=pd.to_numeric(k["kyc_risk_score"])).merge(cust_label, left_on="customer_id", right_index=True)
            if j["label"].nunique() == 2 and len(j) > 200:
                auc = roc_auc_score(j["label"], j["risk"])
                stats["synthetic_kyc_vs_label_auc"] = float(auc)
                R.check("simulated KYC is independent of AML labels (AUC ~ 0.5)", abs(auc - 0.5) < 0.07,
                        f"AUC={auc:.3f}; a larger deviation means the simulation leaks label information", "warning")

    # provenance
    link_types = t["id_mappings"]["link_type"].value_counts().to_dict()
    stats["id_mapping_link_types"] = link_types
    R.check("customer<->account links are documented as synthetic where applicable",
            "SYNTHETIC_ASSOCIATION" in link_types or (t["customers"]["data_origin"] != "SYNTHETIC").all())
    quarantine = PROCESSED_DIR / "quarantine_transactions.csv"
    stats["quarantined_transactions"] = int(max(sum(1 for _ in open(quarantine)) - 1, 0)) if quarantine.exists() else 0

    # missing values
    stats["missing_values"] = {n: {c: int((df[c] == "").sum()) for c in df.columns if (df[c] == "").any()}
                               for n, df in t.items()}

    report = {"generated_at": utc_now_iso(), "passed": not R.errors, "errors": len(R.errors),
              "warnings": len(R.warnings), "statistics": stats, "checks": R.checks}
    write_json(MANIFESTS_DIR / "validation_report.json", report)
    md = ["# Dataset validation report", "", f"Generated: {report['generated_at']}",
          f"Result: **{'PASSED' if report['passed'] else 'FAILED'}** ({len(R.errors)} errors, {len(R.warnings)} warnings)", "",
          "| Check | Result | Detail |", "|---|---|---|"]
    for c in R.checks:
        res = "pass" if c["passed"] else ("FAIL" if c["severity"] == "error" else "warn")
        md.append(f"| {c['name']} | {res} | {c['detail'][:120]} |")
    (MANIFESTS_DIR / "validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for c in R.checks:
        if not c["passed"]:
            print(f"[{c['severity'].upper()}] {c['name']}: {c['detail']}")
    print(f"validation: {len(R.checks)} checks, {len(R.errors)} errors, {len(R.warnings)} warnings")
    return 0 if not R.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
