import pandas as pd


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


def generate_graph(perf, prev, dg):
    nodes_list = perf.customerid
    for n in nodes_list:
        yield {
            "graph_label": perf.loc[perf.customerid == n, "good_bad_flag"].values[0],
            "node_type_perf": perf.loc[perf.customerid == n, :].values.tolist(),
            "node_type_prev": prev.loc[prev.customerid == n, :].values.tolist(),
            "node_type_dg": dg.loc[dg.customerid == n, :].values.tolist()
        }