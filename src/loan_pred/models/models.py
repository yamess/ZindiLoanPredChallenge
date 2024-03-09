from typing import List, Tuple

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalEmbeddingModel(nn.Module):
    def __init__(self, emb_dims: List[Tuple], cont_dim: int, output_size: int, dropout: float = 0.3):
        super(CategoricalEmbeddingModel, self).__init__()

        self.emb_dims = emb_dims
        self.cont_dims = cont_dim

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.cont_dims)

        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(cat, size) for cat, size in self.emb_dims]
        )

        n_emb = sum(emb.embedding_dim for emb in self.embedding_layers)

        self.linears = nn.Sequential(
            nn.Linear(in_features=n_emb + cont_dim, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=output_size)
        )

        # Initialize layers weights
        self.embedding_layers.apply(self.init_layers)
        self.linears.apply(self.init_layers)

    @staticmethod
    def init_layers(m):
        if type(m) == nn.Linear or type(m) == nn.Embedding:
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x_cont, x_emb):
        emb = [
            self.dropout(f(x_emb[:, i])) for i, f in enumerate(self.embedding_layers)
        ]
        emb = torch.cat(emb, 1)

        x_cont = self.bn(x_cont)
        x = torch.cat((emb, x_cont), 1)
        x = self.linears(x)
        return x


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict(
            {
                name: nn.Linear(in_size, out_size) for name in etypes
            }
        )

    def forward(self, g, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            wh = self.weight[etype](feat_dict[srctype])
            g.nodes[srctype].data[f"wh_{etype}"] = wh
            funcs[etype] = (fn.copy_u(f"wh_{etype}", "m"), fn.sum("m", "feat"))
        g.multi_update_all(funcs, "sum")
        return {ntype: g.nodes[ntype].data["feat"] for ntype in g.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, etypes):
        super(HeteroRGCN, self).__init__()
        self.layer1 = HeteroRGCNLayer(in_size=in_size, out_size=in_size, etypes=etypes)
        self.layer2 = HeteroRGCNLayer(in_size=in_size, out_size=hidden_size, etypes=etypes)
        self.classify = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, g):
        feat_dict = g.ndata['feat']
        h = self.layer1(g, feat_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.layer2(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            x = self.classify(hg)
            return x
