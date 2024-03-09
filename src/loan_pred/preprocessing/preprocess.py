import pandas as pd
from pandas.errors import InvalidIndexError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder


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


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.scaler = StandardScaler()
        self.cols = cols

    def fit(self, data):
        _tmp = data[self.cols]
        self.scaler.fit(_tmp)
        return self

    def transform(self, data):
        try:
            _cols = data.columns
            _tmp_1 = data.drop(self.cols, axis=1)
            _tmp_2 = data[self.cols]
            _tmp_2 = self.scaler.transform(_tmp_2)
            _tmp_2 = pd.DataFrame(_tmp_2, columns=self.cols)
            _tmp = pd.concat([_tmp_1, _tmp_2], axis=1)
            _tmp = _tmp[_cols]
            return _tmp
        except InvalidIndexError as e:
            raise e


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoders = {
            c: LabelEncoder() for c in self.cols
        }

    def fit(self, data):
        output = data.copy()
        for c in self.cols:
            self.encoders[c].fit(output[c])
        return self

    def transform(self, data):
        output = data.copy()
        for c in self.cols:
            output.loc[:, c] = self.encoders[c].transform(output[c])
        return output


class TargetEncoder:
    def __init__(self, auto: bool = True, mapping: dict = None):
        self.auto = auto
        self.mapping = mapping
        self._input_validity()

    def encode_target(self, data, target):
        data_ = data.copy()
        if self.auto:
            cat = data.loc[:, target].unique()
            self.mapping = {}
            for i in range(len(cat)):
                self.mapping[cat[i]] = i
        data_["y"] = data_.loc[:, target].apply(lambda x: self.mapping[x])
        data_.drop(target, axis=1, inplace=True)
        data_.rename(columns={"y": target}, inplace=True)
        return data_

    def _input_validity(self):
        if self.auto and self.mapping is not None:
            raise Exception(
                f"Not allowed to set auto=True and provide mapping! Please set auto to False and provide mapping"
            )
