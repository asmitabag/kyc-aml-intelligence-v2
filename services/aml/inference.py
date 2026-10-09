"""GNN inference service (owner: Saloni).

Loads a trained checkpoint, scores accounts from the CURRENT database state, and returns the score, the review
decision, human-readable risk factors and graph evidence. If torch / torch_geometric / the checkpoint are missing
the service reports status "unavailable" instead of crashing, so the health endpoint stays truthful.

gnn_risk_score is a RANKING score in [0, 1] from a class-weighted model. It is not a calibrated probability of
money laundering and is not proof of wrongdoing: it is a signal for human review.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from services.aml import db as DB
from services.aml import features as F
from services.aml.evidence import EvidenceConfig, build_adjacency, graph_evidence


class GNNService:
    def __init__(self, checkpoint_path, db_path=None, evidence_cfg: EvidenceConfig | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.db_path = Path(db_path) if db_path else DB.default_db_path()
        self.cfg = evidence_cfg or EvidenceConfig()
        self.status, self.reason = "unavailable", ""
        self.model = None
        self._cache_sig, self._scores, self._adj, self._tx, self._accounts = None, {}, None, None, None
        try:
            import torch
            from services.aml.gnn import DirectedGraphSAGE
            if not self.checkpoint_path.exists():
                self.reason = f"checkpoint not found: {self.checkpoint_path}"
                return
            ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self.model = DirectedGraphSAGE(**ck["architecture"])
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()
            self.ck = ck
            self.status = "loaded"
        except ImportError as e:
            self.reason = f"torch / torch_geometric not installed: {e}"
        except Exception as e:  # corrupted checkpoint, version mismatch, ...
            self.status, self.reason = "error", f"{type(e).__name__}: {e}"

    @property
    def model_version(self) -> str:
        return self.ck["model_version"] if self.status == "loaded" else "unavailable"

    def _refresh(self) -> None:
        import torch
        from services.aml.gnn import build_graph
        conn = DB.connect(self.db_path)
        try:
            sig = DB.signature(conn)
            if sig == self._cache_sig:
                return
            accounts, tx, kyc = DB.load_accounts(conn), DB.load_transactions(conn), DB.load_kyc(conn)
        finally:
            conn.close()
        feats = F.compute_account_features(tx, accounts["account_id"])
        if self.ck["use_kyc"]:
            feats = feats.join(F.kyc_features(accounts, kyc))
        x = F.apply_scaler(feats, self.ck["scaler"])  # SAME feature names, order and statistics as training
        data = build_graph(x, accounts["account_id"].tolist(), tx)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(data.x, data.edge_index)).numpy()
        self._scores = dict(zip(accounts["account_id"], prob.astype(float)))
        self._tx_counts = (feats["in_count"] + feats["out_count"]).to_dict()
        self._adj, self._tx, self._accounts, self._cache_sig = build_adjacency(tx), tx, accounts, sig

    def score_account(self, account_id: str, include_evidence: bool = True) -> dict:
        if self.status != "loaded":
            raise RuntimeError(f"GNN unavailable: {self.reason}")
        self._refresh()
        if account_id not in self._scores:
            raise KeyError(account_id)
        customer_id = self._accounts.set_index("account_id").loc[account_id, "customer_id"]
        has_tx = self._tx_counts.get(account_id, 0) > 0
        score = float(self._scores[account_id]) if has_tx else None  # no activity -> no score (never fake 0)
        thr = float(self.ck["threshold"])
        factors, evidence = [], None
        if has_tx and include_evidence:
            evidence = graph_evidence(account_id, self._tx, self._scores, self.cfg, self._adj)
            factors = [{"code": f["code"], "description": f["description"], "source": f["source"]}
                       for f in evidence["flags"]]
        if score is not None and score >= thr:
            factors.insert(0, {"code": "HIGH_GNN_SCORE", "source": "GNN",
                               "description": f"GNN score {score:.2f} is at or above the review threshold {thr:.2f}."})
        return {
            "account_id": account_id, "customer_id": customer_id,
            "gnn_risk_score": None if score is None else round(score, 4),
            "review_threshold": round(thr, 4), "flagged_for_review": bool(score is not None and score >= thr),
            "reason": None if has_tx else "NO_TRANSACTIONS",
            "used_kyc_features": bool(self.ck["use_kyc"]), "model_version": self.model_version,
            "dataset_version": self.ck["dataset_version"],
            "score_semantics": "ranking score in [0,1]; not a calibrated probability; a signal for human review",
            "top_risk_factors": factors, "graph_evidence": evidence,
        }
