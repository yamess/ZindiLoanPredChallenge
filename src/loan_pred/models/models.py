from typing import List, Tuple

import torch
import torch.nn as nn


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
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16, out_features=output_size)
        )

        # Initialize layers weights
        self.embedding_layers.apply(self.init_layers)
        self.linears.apply(self.init_layers)

    @staticmethod
    def init_layers(m):
        if type(m) == nn.Linear or type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x_cont, x_emb):
        emb = [
            self.dropout(f(x_emb[:, i])) for i, f in enumerate(self.embedding_layers)
        ]
        emb = torch.cat(emb, 1)

        x_cont = self.bn(x_cont)
        x = torch.cat((emb, x_cont), 1)
        x = self.linears(x)
        return x
