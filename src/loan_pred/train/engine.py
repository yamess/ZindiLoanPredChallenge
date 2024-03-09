import logging
import time

import torch

from loan_pred.train.evaluate import evaluate, evaluate_graph
from src.loan_pred.train.train_model import train, train_graph

logger = logging.Logger(__file__)


def engine(model, train_dataloader, eval_dataloader, optimizer, criterion, scheduler, device, cut_point,
           config, checkpoint, storage_path):
    if checkpoint:
        best_eval_loss = checkpoint["best_eval_loss"]
        best_roc_auc = checkpoint["best_roc_auc"]
        epoch_at_best = checkpoint["epoch_at_best"]
    else:
        checkpoint = {}
        best_eval_loss = 1e5
        best_roc_auc = 0.0
        epoch_at_best = 0

    best_emb_layers = None

    print("======================= Training Started ============================")
    try:
        for e in range(epoch_at_best, config["N_EPOCHS"] + epoch_at_best):
            e_start_time = time.time()

            metrics_train = train(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                cut_point=cut_point
            )
            metrics_eval = evaluate(
                model=model,
                dataloader=eval_dataloader,
                criterion=criterion,
                device=device,
                cut_point=cut_point
            )

            if scheduler:
                scheduler.step(metrics_eval["avg_loss"])

            e_end_time = time.time()
            e_elapsed_time = e_end_time - e_start_time

            display_msg = (
                f"Epoch: {e: <{4}} | Elapsed Time: {e_elapsed_time: 3.2f} s | Train Loss: {metrics_train['avg_loss']: .4f} "
                f"| Valid Loss: {metrics_eval['avg_loss']: .4f} | Train ROC AUC: {metrics_train['roc_auc']: .4f} | "
                f"Valid ROC AUC: {metrics_eval['roc_auc']: .4f} | "
            )

            if metrics_eval["avg_loss"] < best_eval_loss:
                best_eval_loss = metrics_eval["avg_loss"]
                best_roc_auc = metrics_eval["roc_auc"]
                best_state_dict = model.state_dict()
                epoch_at_best = e
                best_emb_layers = model.embedding_layers

                display_msg += " + "

                checkpoint["best_eval_loss"] = best_eval_loss
                checkpoint["best_roc_auc"] = best_roc_auc
                checkpoint["epoch_at_best"] = epoch_at_best
                checkpoint["model_state_dict"] = best_state_dict
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            print(display_msg)
    except Exception as er:
        logging.error(er)
    finally:
        torch.save(best_emb_layers, "../models_storage/embeddings/embeddings_layers.pt")
        torch.save(checkpoint, storage_path)


def engine_graph(model, trainloader, testloader, optimizer, criterion, n_epoch, device):
    for e in range(n_epoch):
        train_res = train_graph(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=trainloader,
            device=device
        )
        test_res = evaluate_graph(
            model=model,
            criterion=criterion,
            dataloader=testloader,
            device=device
        )
        print(test_res)
