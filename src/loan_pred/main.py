import random

import torch
from dgl.dataloading import GraphDataLoader
from torch import nn, optim

from loan_pred.config import config
from loan_pred.helpers.helper import load_pickle
from loan_pred.models.models import HeteroRGCN
from loan_pred.preprocessing.embedding import EmbeddingTransformer
from loan_pred.preprocessing.get_data import get_train_dg, get_train_loans, get_train_prevloans
from loan_pred.preprocessing.graph_processing import generate_graphs_iterator, HeteroGraphDataset
from loan_pred.train.engine import engine_graph

if __name__ == "__main__":
    # dg data prep
    scaler_dg = load_pickle(file_path="../../models_storage/scalers/dg_scaler.pk")
    encoder_dg = load_pickle(file_path="../../models_storage/encoders/dg_multilabel_encoder.pk")
    train_dg = get_train_dg(
        path="../../data/preprocessed/train/train_dg.csv",
        encoder=encoder_dg,
        scaler=scaler_dg
    )

    # perf data prep
    train_perf = get_train_loans(
        path="../../data/preprocessed/train/train_perf.csv",
        encoder=load_pickle(file_path="../../models_storage/encoders/loan_target_encoder"),
        scaler=load_pickle(file_path="../../models_storage/scalers/loan_scaler.pk")
    )

    # prev loans data prep
    train_prevloans = get_train_prevloans(
        path="../../data/preprocessed/train/train_prevloans.csv",
        scaler=load_pickle(file_path="../../models_storage/scalers/prevloan_scaler.pk")
    )

    # Apply embedding transformation
    embeddings_weight = load_pickle(file_path="../../models_storage/embeddings/embeddings_weights.pk")
    embedder = EmbeddingTransformer(embedding_weights=embeddings_weight)
    train_dg = embedder.transform(train_dg)

    # Split data
    n = len(train_perf)
    train_len = int(n * 0.8)
    test_len = n - train_len
    population = list(range(n))

    train_mask = random.sample(population, train_len)
    test_mask = [k for k in population if k not in train_mask]

    train_data = train_perf.loc[train_mask]
    test_data = train_perf.loc[test_mask]

    # graph iterator
    train_graph_generator = generate_graphs_iterator(
        loan=train_data,
        dg=train_dg,
        prev_loan=train_prevloans
    )
    test_graph_generator = generate_graphs_iterator(
        loan=test_data,
        dg=train_dg,
        prev_loan=train_prevloans
    )

    # Generate graph dataset
    train_dataset = HeteroGraphDataset(train_graph_generator)
    train_dataloader = GraphDataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = HeteroGraphDataset(test_graph_generator)
    test_dataloader = GraphDataLoader(test_dataset, batch_size=50)

    for batch, label in train_dataloader:
        try:
            de = batch
            label = label
        except Exception as er:
            print(er)

    pos_weight = train_data.good_bad_flag.value_counts()[0] / train_data.good_bad_flag.value_counts()[1]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    criterion.to(config["DEVICE"])

    model = HeteroRGCN(in_size=5, hidden_size=5, out_size=1, etypes=['has', 'lives']).to(config["DEVICE"])

    optimizer = optim.Adam(model.parameters(), config["LR"])

    engine_graph(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        trainloader=train_dataloader,
        testloader=test_dataloader,
        device=config["DEVICE"],
        n_epoch=2
    )
