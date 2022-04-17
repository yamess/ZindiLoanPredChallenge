import numpy as np
import torch


def generate_graphs_data(loan, prev_loan, dg):
    node_list = loan.customerid
    for k, n in enumerate(node_list):
        loan_row = loan.loc[loan.customerid == n, :].reset_index(drop=True)
        dg_rows = dg.loc[dg.customerid == n, :].reset_index(drop=True)
        prev_rows = prev_loan.loc[prev_loan.customerid == n, :].reset_index(drop=True)
        if len(dg_rows) > 0:
            dob = dg.loc[dg.customerid == n, "birthdate"].reset_index(drop=True)
            loan_row.loc[:, "age_at_loan"] = (
                                                     (loan_row["approveddate"] - dob) / np.timedelta64(1,
                                                                                                       "Y")
                                             ) / 50  # To normalize the age

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


def get_node_feat(data):
    feat_n = {
        "node_loans": torch.tensor(data["node_loans"]),
        "node_dg": torch.tensor(data["node_dg"]),
        "node_prevloans": torch.tensor(data["node_prevloans"])
    }
    return feat_n
