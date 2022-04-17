import numpy as np
from dgl.data import DGLDataset
import dgl


def generate_graphs_iterator(loan, prev_loan, dg):
    node_list = loan.customerid
    for k, n in enumerate(node_list):
        loan_row = loan.loc[loan.customerid == n, :].reset_index(drop=True)
        dg_rows = dg.loc[dg.customerid == n, :].reset_index(drop=True)
        prev_rows = prev_loan.loc[prev_loan.customerid == n, :].reset_index(drop=True)
        if len(dg_rows) > 0:
            dob = dg.loc[dg.customerid == n, "birthdate"].reset_index(drop=True)
            loan_row.loc[:, "age_at_loan"] = ((loan_row["approveddate"] - dob) / np.timedelta64(1, "Y")) / 50
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
            "node_dg": dg_rows,
            "node_prevloans": prev_rows
        }
        yield graph


# def get_node_feat(data):
#     feat_n = {
#         "node_loans": torch.tensor(data["node_loans"]),
#         "node_dg": torch.tensor(data["node_dg"]),
#         "node_prevloans": torch.tensor(data["node_prevloans"])
#     }
#     return feat_n


class HeteroGraphDataset(DGLDataset):
    def __init__(self, data_iterator):
        super(HeteroGraphDataset, self).__init__()
        self.data = data_iterator
        self.graph = None

    def process(self):
        data = {}
        for d in 
        for node in ["node_prevloans", "node_dg"]:
            node_ids = list(range(len(data[node])))
            if len(node_ids) > 0:
                if node == "node_prevloans":
                    rel = "has"
                else:
                    rel = "lives"
                test_[("node_loans", rel, node)] = (np.array([0] * len(node_ids)), np.array(node_ids))
        self.graph =

    def __len__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented
