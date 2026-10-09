import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

from tests.support import built_dataset


@unittest.skipUnless(HAVE_TORCH, "torch / torch_geometric not installed")
class GnnSmokeTests(unittest.TestCase):
    def test_train_checkpoint_and_inference_round_trip(self):
        from services.aml.inference import GNNService
        from services.aml.train_gnn import train
        root = built_dataset()
        rep, art = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        res = train("t_gnn", False, epochs=30, hidden=16, layers=2, lr=0.01, dropout=0.2, patience=10, seed=1,
                    root=root, report_dir=rep, artifact_dir=art)
        self.assertIn("pr_auc", res["test"])
        svc = GNNService(art / "t_gnn" / "checkpoint.pt", root / "db" / "kycaml.sqlite")
        self.assertEqual(svc.status, "loaded", svc.reason)
        out = svc.score_account("A000001")
        self.assertTrue(0.0 <= out["gnn_risk_score"] <= 1.0)
        self.assertEqual(out["model_version"], "gnn-v1")

    def test_kyc_aware_checkpoint_rejects_missing_features(self):
        from services.aml.features import apply_scaler
        import pandas as pd
        with self.assertRaises(ValueError):
            apply_scaler(pd.DataFrame({"a": [1.0]}), {"feature_names": ["a", "kyc_risk_score"], "mean": [0, 0], "std": [1, 1]})


if __name__ == "__main__":
    unittest.main()
