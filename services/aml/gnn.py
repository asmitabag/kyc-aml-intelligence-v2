"""Directed GraphSAGE model + graph construction (owner: Saloni). Requires torch and torch_geometric."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class DirectedGraphSAGE(nn.Module):
    """Money flows have direction, so every layer aggregates incoming and outgoing neighbours with separate
    weights (who sends to me vs. whom I send to) and adds them to the node's own transformed state.
    A laundering intermediary has a different in/out neighbourhood signature from a merchant or payroll account.
    """

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.conv_in = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(layers)])
        self.conv_out = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.dropout = dropout

    def embed(self, x, edge_index):
        rev = edge_index.flip(0)
        for conv_in, conv_out, norm in zip(self.conv_in, self.conv_out, self.norms):
            h = conv_in(x, edge_index) + conv_out(x, rev)  # messages flowing in + messages flowing out
            x = Fn.dropout(Fn.relu(norm(h)), p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index):
        return self.head(self.embed(x, edge_index)).squeeze(-1)  # logits


def build_graph(x: np.ndarray, account_ids: list[str], tx: pd.DataFrame, y: np.ndarray | None = None) -> Data:
    """Accounts are nodes; each transaction is a directed edge from sender to receiver.
    Edge attributes (log amount) are stored for evidence/visualisation; SAGEConv itself uses topology + node features."""
    index = {a: i for i, a in enumerate(account_ids)}
    src = tx["from_account_id"].map(index).to_numpy()
    dst = tx["to_account_id"].map(index).to_numpy()
    keep = ~(pd.isna(src) | pd.isna(dst))
    edge_index = torch.tensor(np.vstack([src[keep], dst[keep]]).astype("int64"))
    data = Data(x=torch.from_numpy(x), edge_index=edge_index)
    data.edge_attr = torch.tensor(np.log1p(pd.to_numeric(tx["amount"]).to_numpy()[keep]).astype("float32")).unsqueeze(1)
    if y is not None:
        data.y = torch.tensor(y.astype("float32"))
    data.account_ids = account_ids
    return data
