from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class CreateTensorDataset(Dataset):
    def __init__(self, emb_cols: List[str], x_data, y_data):
        super(CreateTensorDataset, self).__init__()
        self.emb_cols = emb_cols

        # Embedding data
        emb_ = x_data.loc[:, self.emb_cols]
        self.emb_data = np.stack(
            [c.values for _, c in emb_.items()], axis=1
        ).astype(np.int64)

        # Continuous data
        other_cols = emb_cols
        other_data = x_data.drop(other_cols, axis=1)
        self.cont_data = np.stack(
            [c.values for _, c in other_data.items()], axis=1
        ).astype(np.float32)
        self.y = y_data.values.astype(np.int32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_cont = self.cont_data[item]
        x_emb = self.emb_data[item]
        y = np.asarray(self.y[item])

        out = {
            "x_cont": torch.from_numpy(x_cont),
            "x_emb": torch.from_numpy(x_emb),
            "y": torch.tensor(y, dtype=torch.long)
        }
        return out


class CategoricalEmbeddingSizes:
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols

    def get_emb_cols(self, data):
        emb_cols = [
            c for c in self.cat_cols if len(data[c].unique()) > 2
        ]
        return emb_cols

    def get_cat_emb_dims(self, data):
        emb_cols = self.get_emb_cols(data=data)
        cat_dims = [len(data[col].unique()) + 1 for col in emb_cols]
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
        return emb_dims, emb_cols
