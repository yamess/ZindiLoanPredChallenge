import pandas as pd
import numpy as np

from src.loan_pred.preprocessing.preprocess import get_data, generate_graph

if __name__ == "__main__":

    loan, prevloan, dg = get_data(
        loan_path="../../data/preprocessed/train/train_perf.csv",
        prev_loan_path="../../data/preprocessed/train/train_prevloans.csv",
        dg_path="../../data/preprocessed/train/train_dg.csv"
    )

    generator = generate_graph(
        perf=loan,
        prev_loan=prevloan,
        dg=dg
    )
    print(next(generator))
