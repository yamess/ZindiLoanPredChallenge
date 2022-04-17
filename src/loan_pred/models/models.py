from typing import List, Tuple

import dgl
import dgl.nn.pytorch as dglnn
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


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, rel_name):
        super(RGCN, self).__init__()

        self.conv1 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_feats=in_feats, out_feats=hid_feats) for rel in rel_name},
            aggregate="sum"
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_feats=in_feats, out_feats=hid_feats) for rel in rel_name},
            aggregate="sum"
        )

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h['node_loans'] = inputs['node_loans']
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super(HeteroClassifier, self).__init__()

        self.rgcn = RGCN(in_feats=in_dim, hid_feats=hidden_dim, rel_name=rel_names)
        self.classify = nn.Linear(in_features=hidden_dim, out_features=n_classes)

    def forward(self, g):
        h = g.ndata["feat"]
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata["h"] = h
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, "h", ntype=ntype)
            x = self.classify(hg)
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

