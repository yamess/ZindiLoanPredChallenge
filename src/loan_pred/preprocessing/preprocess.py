import pandas as pd
import numpy as np


def convert_dtype(data: pd.DataFrame, columns_type: dict) -> pd.DataFrame:
    """
    Function to convert pandas dataframe columns to a specific type except the datetime.

    :param data: pandas dataframe
    :param columns_type: dict, dictionary where the k in the column name and the value is the type
    :return: pandas dataframe, tranformed data
    """
    data = data.copy()
    for c, t in columns_type.items():
        if t.lower().startswith("date"):
            data[c] = pd.to_datetime(data[c], infer_datetime_format=True)
        else:
            data[c] = data[c].astype(t)
    return data


# def generate_graph(perf, prev, dg):
#     nodes_list = perf.customerid
#     for n in nodes_list:
#         yield {
#             "graph_label": perf.loc[perf.customerid == n, "good_bad_flag"].values[0],
#             "node_type_perf": perf.loc[perf.customerid == n, :].values.tolist(),
#             "node_type_prev": prev.loc[prev.customerid == n, :].values.tolist(),
#             "node_type_dg": dg.loc[dg.customerid == n, :].values.tolist()
#         }


def get_data(loan_path, prev_loan_path, dg_path):
    train_prevloans = pd.read_csv(prev_loan_path)
    cols_dtypes = {
        "customerid": "category",
        "loannumber": "int",
        "loanamount": "float",
        "totaldue": "float",
        "termdays": "int",
        "closeddate_days": "int",
        "firstduedate_days": "int",
        "firstrepaiddate_days": "int",
    }
    train_prevloans = convert_dtype(data=train_prevloans, columns_type=cols_dtypes)

    train_dg = pd.read_csv(dg_path)
    cols_dtypes = {
        "customerid": "category",
        "birthdate": "datetime",
        "bank_account_type": "category",
        "longitude_gps": "float",
        "latitude_gps": "float",
        "bank_name_clients": "category",
        "employment_status_clients": "category",
        "is_missing_emp_status_clients": "int"
    }
    train_dg = convert_dtype(data=train_dg, columns_type=cols_dtypes)
    train_dg.head()

    train_perf = pd.read_csv(loan_path)
    cols_dtypes = {
        "customerid": "category",
        "loannumber": "int",
        "approveddate": "datetime",
        "loanamount": "float",
        "totaldue": "float",
        "termdays": "int",
        "good_bad_flag": "category"
    }
    train_perf = convert_dtype(data=train_perf, columns_type=cols_dtypes)
    train_perf.head()

    return train_perf, train_prevloans, train_dg


def generate_graph(perf, prev_loan, dg):
    nodes_list = perf.customerid
    prevloans_cols = ["loannumber", "loanamount", "totaldue", "termdays", "closeddate_days", "firstduedate_days", "firstrepaiddate_days"]
    dg_cols = ["bank_account_type","longitude_gps", "latitude_gps", "bank_name_clients", "employment_status_clients", "is_missing_emp_status_clients"]

    for n in nodes_list:
        loan_row = perf.loc[perf.customerid == n, :].reset_index(drop=True)
        if len(dg[dg.customerid == n]) > 0:
            dob = dg.loc[dg.customerid == n, "birthdate"].reset_index(drop=True)

            loan_row.loc[:, "age_at_loan"] = ((loan_row["approveddate"] - dob) / np.timedelta64(1, "Y"))

            loans_cols = ["loannumber", "loanamount", "totaldue", "termdays", "age_at_loan"]

            response = {
                "user_id": n,
                "graph_label": loan_row.good_bad_flag[0],
                "node_type_loans": loan_row.loc[:, loans_cols].values.tolist(),
                "node_type_prevloans": prev_loan.loc[prev_loan.customerid == n, prevloans_cols].values.tolist(),
                "node_type_dg": dg.loc[dg.customerid == n, dg_cols].values.tolist()
            }
        else:
            loans_cols = ["loannumber", "loanamount", "totaldue", "termdays"]
            response = {
                "user_id": n,
                "graph_label": loan_row.good_bad_flag[0],
                "node_type_loans": loan_row.loc[loan_row.customerid == n, loans_cols].values.tolist(),
                "node_type_prevloans": prev_loan.loc[prev_loan.customerid == n, prevloans_cols].values.tolist(),
                "node_type_dg": dg.loc[dg.customerid == n, dg_cols].values.tolist()
            }
        yield response