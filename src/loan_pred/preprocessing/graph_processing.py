import dgl
import numpy as np
import torch
from dgl import DGLError
from dgl.data import DGLDataset


def generate_graphs_iterator(loan, prev_loan, dg):
    node_list = loan.customerid
    for k, n in enumerate(node_list):
        loan_row = loan.loc[loan.customerid == n, :].reset_index(drop=True)
        dg_rows = dg.loc[dg.customerid == n, :].reset_index(drop=True)
        prev_rows = prev_loan.loc[prev_loan.customerid == n, :].reset_index(drop=True)
        if len(dg_rows) > 0:
            dob = dg.loc[dg.customerid == n, "birthdate"].reset_index(drop=True)
            loan_row.loc[:, "age_at_loan"] = ((loan_row["approveddate"] - dob) / np.timedelta64(1, "Y")) / 50
        else:
            loan_row.loc[:, "age_at_loan"] = 20 / 50
            # To normalize the age
        label = loan_row.good_bad_flag[0]
        loan_row = loan_row.drop(["customerid", "approveddate", "good_bad_flag"], axis=1).values.tolist()
        dg_rows = dg_rows.drop(["customerid", "birthdate"], axis=1).values.tolist()
        prev_rows = prev_rows.drop("customerid", axis=1).values.tolist()

        graph = {
            "id": k,
            "user_id": n,
            "label": label,
            "node_loans": loan_row,
            "node_dg": dg_rows if dg_rows != [] else [[0.0] * 19],
            "node_prevloans": prev_rows if prev_rows != [] else [[0.0] * 11]
        }
        yield graph


class HeteroGraphDataset(DGLDataset):
    def __init__(self, data_iterator):
        # super().__init__()
        self.data = data_iterator
        self.graphs = None
        self.labels = None
        self.process()

    @classmethod
    def process_single_graph(cls, data):
        tmp = {}
        for node in ["node_prevloans", "node_dg"]:
            node_ids = list(range(len(data[node])))
            rel = "has" if node == "node_prevloans" else "lives"
            tmp[("node_loans", rel, node)] = (np.array([0] * len(node_ids)), np.array(node_ids))
        return tmp

    def process(self):
        graph_ = []
        labels = []
        for k, v in enumerate(self.data):
            tmp = HeteroGraphDataset.process_single_graph(v)
            tmp_graph = dgl.heterograph(tmp)
            try:
                tmp_graph.nodes["node_loans"].data["feat"] = torch.tensor(
                    v["node_loans"],
                    dtype=torch.float32
                )
                tmp_graph.nodes["node_dg"].data["feat"] = torch.tensor(
                    v["node_dg"],
                    dtype=torch.float32
                )
                tmp_graph.nodes["node_prevloans"].data["feat"] = torch.tensor(
                    v["node_prevloans"],
                    dtype=torch.float32
                )

                graph_.append(tmp_graph)
                labels.append(torch.tensor(v["label"]))
            except DGLError as er:
                print(er)
        self.graphs = graph_.copy()
        self.labels = labels.copy()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item]
