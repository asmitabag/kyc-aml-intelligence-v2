import json
import sqlite3
import unittest

import numpy as np
import pandas as pd

from tests.support import REPO, built_dataset


class DatasetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = built_dataset()
        cls.proc = cls.root / "processed"

    def test_validation_passed_and_manifest_written(self):
        report = json.loads((self.root / "manifests" / "validation_report.json").read_text())
        self.assertTrue(report["passed"], [c for c in report["checks"] if not c["passed"] and c["severity"] == "error"])
        manifest = json.loads((self.root / "manifests" / "dataset_manifest.json").read_text())
        self.assertEqual(manifest["dataset_version"], "dataset_v1")
        self.assertIn("SYNTHETIC_ASSOCIATION", json.dumps(manifest["identity_links"]))

    def test_database_foreign_keys_hold(self):
        conn = sqlite3.connect(self.root / "db" / "kycaml.sqlite")
        conn.execute("PRAGMA foreign_keys = ON")
        self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0], 1000)

    def test_splits_disjoint_cover_and_keep_groups_whole(self):
        accounts = set(pd.read_csv(self.proc / "accounts.csv", dtype=str)["account_id"])
        parts = {s: pd.read_csv(self.root / "splits" / f"{s}_ids.csv", dtype=str) for s in ("train", "validation", "test")}
        ids = [set(p["entity_id"]) for p in parts.values()]
        self.assertEqual(set().union(*ids), accounts)
        self.assertEqual(sum(len(i) for i in ids), len(accounts))
        groups = [set(p["group_id"]) for p in parts.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_canonical_ids_carry_no_label_information(self):
        labels = pd.read_csv(self.proc / "aml_labels.csv", dtype={"entity_id": str})
        acct = labels[labels["entity_type"] == "ACCOUNT"].copy()
        acct["n"] = acct["entity_id"].str[1:].astype(int)
        rho = acct[["n", "label"]].corr(method="spearman").iloc[0, 1]
        self.assertLess(abs(rho), 0.08, f"account number correlates with label: rho={rho:.3f}")

    def test_aml_labels_are_not_in_kyc_generator(self):
        src = (REPO / "data" / "scripts" / "build_kyc_features.py").read_text()
        code = "\n".join(line for line in src.splitlines() if not line.strip().startswith(("#", '"""')))
        self.assertNotIn("read_csv(PROCESSED_DIR / \"aml_labels", code)
        self.assertNotIn("is_suspicious_label", code.split("def status")[1])

    def test_customer_account_links_marked_synthetic(self):
        m = pd.read_csv(self.proc / "id_mappings.csv", dtype=str)
        links = m[m["canonical_entity_type"] == "CUSTOMER"]
        self.assertTrue((links["link_type"] == "SYNTHETIC_ASSOCIATION").all())
        self.assertGreater(len(links), 0)

    def test_no_fabricated_biometric_records(self):
        for name in ("liveness_manifest.csv", "document_manifest.csv"):
            self.assertEqual(len(pd.read_csv(self.proc / name)), 0)


class FeatureLeakageTests(unittest.TestCase):
    def test_features_ignore_label_column(self):
        from services.aml.features import compute_account_features
        built_dataset()
        root = built_dataset()
        tx = pd.read_csv(root / "processed" / "transactions.csv", dtype=str)
        ids = pd.read_csv(root / "processed" / "accounts.csv", dtype=str)["account_id"]
        with_label = compute_account_features(tx, ids)
        shuffled = tx.copy()
        shuffled["is_suspicious_label"] = np.random.default_rng(0).permutation(shuffled["is_suspicious_label"].to_numpy())
        pd.testing.assert_frame_equal(with_label, compute_account_features(shuffled, ids))
        no_label = compute_account_features(tx.drop(columns=["is_suspicious_label"]), ids)
        pd.testing.assert_frame_equal(with_label, no_label)

    def test_scaler_uses_only_the_rows_it_is_given(self):
        from services.aml.features import fit_scaler
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
        s = fit_scaler(df)
        self.assertEqual(s["mean"], [2.0, 5.0])
        self.assertEqual(s["std"][1], 1.0)  # constant column does not divide by zero


class BaselineTests(unittest.TestCase):
    def test_baseline_beats_chance_and_reports_all_metrics(self):
        import tempfile
        from pathlib import Path
        from services.aml.train_baseline import run
        root = built_dataset()
        res = run("t_baseline", False, 1, root, Path(tempfile.mkdtemp(prefix="kycaml_rep_")))
        t = res["test"]
        for k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "confusion_matrix"):
            self.assertIn(k, t)
        self.assertGreater(t["pr_auc"], t["support"]["prevalence"])  # better than a random scorer
        self.assertGreater(t["roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
