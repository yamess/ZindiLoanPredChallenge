import pandas as pd

from src.loan_pred.preprocessing.preprocess import convert_dtype, CustomScaler, TargetEncoder, MultiLabelEncoder


def get_train_loans(path: str, scaler: CustomScaler, encoder: TargetEncoder):
    cols_dtypes = {
        "customerid": "category",
        "loannumber": "int",
        "approveddate": "datetime",
        "loanamount": "float",
        "totaldue": "float",
        "termdays": "int",
        "good_bad_flag": "category"
    }
    _tmp = pd.read_csv(path)
    _tmp = convert_dtype(data=_tmp, columns_type=cols_dtypes)
    _tmp = scaler.transform(_tmp)
    _tmp = encoder.encode_target(_tmp, target="good_bad_flag")
    return _tmp


def get_train_prevloans(path: str, scaler: CustomScaler):
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
    _tmp = pd.read_csv(path)
    _tmp = convert_dtype(data=_tmp, columns_type=cols_dtypes)
    _tmp = scaler.transform(_tmp)
    return _tmp


def get_train_dg(path: str, encoder: MultiLabelEncoder, scaler: CustomScaler):
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
    _tmp = pd.read_csv(path)
    _tmp = convert_dtype(data=_tmp, columns_type=cols_dtypes)
    _tmp = encoder.transform(_tmp)
    _tmp = scaler.transform(_tmp)
    return _tmp
