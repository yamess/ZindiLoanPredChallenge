import logging
import time
from copy import deepcopy

import torch

from src.loan_pred.train.train_model import train, evaluate

logger = logging.Logger(__file__)


def engine(model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, config, checkpoint):
    best_state_dict = None
    if checkpoint:
        best_eval_loss = checkpoint["best_eval_loss"]
        best_roc_auc = checkpoint["best_roc_auc"]
        epoch_at_best = checkpoint["epoch_at_best"]
    else:
        checkpoint = {}
        best_eval_loss = 1e5
        best_roc_auc = 0.0
        epoch_at_best = 0

    print("======================= Training Started ============================")

    for e in range(config["N_EPOCHS"]):
        e_start_time = time.time()

        metrics_train = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=config["DEVICE"],
            cut_point=config["cut_point"]
        )
        metrics_eval = evaluate(
            model=model,
            dataloader=eval_dataloader,
            criterion=criterion,
            device=config["DEVICE"],
            cut_point=config["cut_point"]
        )

        if scheduler:
            scheduler.step(metrics_eval["avg_loss"])

        e_end_time = time.time()
        e_elapsed_time = e_end_time - e_start_time

        display_msg = (
            f"Epoch: {e: <{4}} | Elapsed Time: {e_elapsed_time: 3.2f} s | Train Loss: {metrics_train['avg_loss']: .4f} "
            f"| Valid Loss: {metrics_eval['avg_loss']: .4f} | Train ROC AUC: {metrics_train['roc_auc']: .4f} | "
            f"Valid ROC AUC: {metrics_eval['roc_auc']: .4f} | Train MBE: {metrics_train['mbe']: .4f} | "
        )

        if (metrics_eval["avg_loss"] < best_eval_loss) & (metrics_eval["roc_auc"] >= best_roc_auc):
            best_eval_loss = metrics_eval["avg_loss"]
            best_roc_auc = metrics_eval["roc_auc"]
            best_state_dict = deepcopy(model.state_dict())
            epoch_at_best = e

            display_msg += " + "

        checkpoint["best_eval_loss"] = best_eval_loss
        checkpoint["best_roc_auc"] = best_roc_auc
        checkpoint["best_state_dict"] = best_state_dict
        checkpoint["epoch_at_best"] = epoch_at_best

        torch.save(checkpoint, config["CHECKPOINT_POINT"])

        print(display_msg)
