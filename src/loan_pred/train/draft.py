import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn


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
