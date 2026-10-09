"""Graph evidence for an account (owner: Saloni). Pure pandas/python, no torch.

IMPORTANT: this is rule-based EVIDENCE computed from stored transactions, shown next to the GNN score so a
reviewer can check it against real transaction IDs. It is NOT an explanation of why the GNN scored an account
(that is GNNExplainer, which Sounak owns). Thresholds below are documented heuristics, not validated detectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class EvidenceConfig:
    rapid_window_hours: float = 24.0   # outgoing within this long after an incoming transfer...
    rapid_min_retention: float = 0.80  # ...forwarding at least this share of the amount (and not more than received)
    fan_window_hours: float = 24.0
    fan_min_counterparties: int = 5    # distinct counterparties inside one window
    cycle_max_len: int = 5             # directed, time-ordered cycles through the account, 3..max length
    cycle_search_budget: int = 20000   # DFS node-visit cap so hub accounts cannot stall the service
    neighbor_score_threshold: float = 0.5
    max_edges: int = 60                # edges returned for visualisation


def build_adjacency(tx: pd.DataFrame) -> dict:
    """account -> list of (neighbour, tx_id, timestamp, amount) for outgoing edges."""
    adj: dict = {}
    for tid, a, b, amt, ts in zip(tx["transaction_id"], tx["from_account_id"], tx["to_account_id"],
                                  tx["amount"], tx["timestamp"]):
        adj.setdefault(a, []).append((b, tid, ts, float(amt)))
    return adj


def _find_cycles(adj: dict, start: str, cfg: EvidenceConfig, max_cycles: int = 3) -> list[dict]:
    found, budget = [], [cfg.cycle_search_budget]

    def dfs(node, path_nodes, path_tx, last_ts):
        if len(found) >= max_cycles or budget[0] <= 0:
            return
        for nxt, tid, ts, _ in adj.get(node, []):
            budget[0] -= 1
            if budget[0] <= 0:
                return
            if ts < last_ts:  # money flows forward in time
                continue
            if nxt == start and len(path_tx) + 1 >= 3:
                found.append({"accounts": path_nodes[:], "tx_ids": path_tx + [tid]})
                if len(found) >= max_cycles:
                    return
            elif nxt not in path_nodes and len(path_nodes) < cfg.cycle_max_len:
                dfs(nxt, path_nodes + [nxt], path_tx + [tid], ts)

    dfs(start, [start], [], pd.Timestamp.min.tz_localize("UTC"))
    return found


def graph_evidence(account_id: str, tx: pd.DataFrame, scores: dict | None = None, cfg: EvidenceConfig | None = None,
                   adjacency: dict | None = None) -> dict:
    """tx columns: transaction_id, from_account_id, to_account_id, amount, timestamp (tz-aware UTC)."""
    cfg = cfg or EvidenceConfig()
    inc = tx[tx["to_account_id"] == account_id].sort_values("timestamp")
    out = tx[tx["from_account_id"] == account_id].sort_values("timestamp")
    flags = []

    # 1. rapid pass-through: incoming followed quickly by outgoing of nearly the same amount
    window = pd.Timedelta(hours=cfg.rapid_window_hours)
    pairs = []
    in_t, in_a, in_id = inc["timestamp"].to_numpy(), inc["amount"].to_numpy(dtype=float), inc["transaction_id"].to_numpy()
    for tid, ts, amt in zip(out["transaction_id"].head(500), out["timestamp"].head(500), out["amount"].head(500)):
        mask = (inc["timestamp"] < ts) & (inc["timestamp"] >= ts - window)
        for k in mask.to_numpy().nonzero()[0]:
            if cfg.rapid_min_retention * in_a[k] <= float(amt) <= in_a[k]:
                pairs.append((in_id[k], tid))
                break
    if pairs:
        flags.append({
            "code": "RAPID_FUND_MOVEMENT", "source": "GRAPH_EVIDENCE",
            "description": f"{len(pairs)} incoming transfer(s) were forwarded within {cfg.rapid_window_hours:g}h, "
                           f"keeping at least {cfg.rapid_min_retention:.0%} of the amount.",
            "tx_ids": sorted({t for p in pairs for t in p})[:40]})

    # 2. fan-out / fan-in: many distinct counterparties inside one window
    for direction, df, col, code, word in (("out", out, "to_account_id", "FAN_OUT", "sent funds to"),
                                           ("in", inc, "from_account_id", "FAN_IN", "received funds from")):
        d = df.tail(2000).reset_index(drop=True)
        best, best_ids, j = 0, [], 0
        for i in range(len(d)):
            while d["timestamp"].iloc[i] - d["timestamp"].iloc[j] > pd.Timedelta(hours=cfg.fan_window_hours):
                j += 1
            seg = d.iloc[j:i + 1]
            n = seg[col].nunique()
            if n > best:
                best, best_ids = n, seg["transaction_id"].tolist()
        if best >= cfg.fan_min_counterparties:
            flags.append({"code": code, "source": "GRAPH_EVIDENCE",
                          "description": f"Account {word} {best} distinct accounts within {cfg.fan_window_hours:g}h.",
                          "tx_ids": best_ids[:40]})

    # 3. circular flow
    adj = adjacency if adjacency is not None else build_adjacency(tx)
    cycles = _find_cycles(adj, account_id, cfg)
    if cycles:
        flags.append({"code": "CIRCULAR_FLOW", "source": "GRAPH_EVIDENCE",
                      "description": f"Funds can be traced around {len(cycles)} time-ordered loop(s) of "
                                     f"{min(len(c['tx_ids']) for c in cycles)}-{max(len(c['tx_ids']) for c in cycles)} "
                                     "transfers returning to this account.",
                      "tx_ids": sorted({t for c in cycles for t in c["tx_ids"]})[:40]})

    # 4. neighbours the model scores as high risk
    if scores:
        nbrs = set(inc["from_account_id"]) | set(out["to_account_id"])
        hot = sorted([n for n in nbrs if scores.get(n, 0.0) >= cfg.neighbor_score_threshold])
        if hot:
            flags.append({"code": "HIGH_RISK_NEIGHBORS", "source": "GNN",
                          "description": f"{len(hot)} directly connected account(s) have GNN score >= "
                                         f"{cfg.neighbor_score_threshold:g}.", "account_ids": hot[:20], "tx_ids": []})

    # subgraph for visualisation: biggest incident edges + evidence transactions
    ev_ids = {t for f in flags for t in f.get("tx_ids", [])}
    incident = pd.concat([inc, out]).drop_duplicates("transaction_id")
    keep = pd.concat([incident[incident["transaction_id"].isin(ev_ids)],
                      incident.sort_values("amount", ascending=False)]).drop_duplicates("transaction_id").head(cfg.max_edges)
    extra = tx[tx["transaction_id"].isin(ev_ids - set(keep["transaction_id"]))].head(cfg.max_edges)
    edges = pd.concat([keep, extra]).drop_duplicates("transaction_id").head(cfg.max_edges)
    nodes = sorted(set(edges["from_account_id"]) | set(edges["to_account_id"]) | {account_id})
    return {
        "account_id": account_id,
        "flags": flags,
        "subgraph": {
            "nodes": [{"account_id": n, "gnn_score": (None if not scores else round(float(scores.get(n, 0.0)), 4)),
                       "is_target": n == account_id} for n in nodes],
            "edges": [{"transaction_id": r.transaction_id, "from_account_id": r.from_account_id,
                       "to_account_id": r.to_account_id, "amount": float(r.amount),
                       "timestamp": r.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "in_evidence": r.transaction_id in ev_ids} for r in edges.itertuples()],
        },
    }
