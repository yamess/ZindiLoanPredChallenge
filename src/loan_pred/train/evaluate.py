import torch
from sklearn.metrics import roc_auc_score, precision_score


def evaluate(model, dataloader, criterion, device, cut_point: float):
    y_trues = []
    losses = 0.0
    probs = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x_emb = batch["x_emb"].to(device)
            x_cont = batch["x_cont"].to(device)
            y_true = batch["y"].float().to(device)

            logits = model(x_emb=x_emb, x_cont=x_cont)
            loss = criterion(logits.squeeze(1), y_true)

            losses += (loss.item() * y_true.size(0))
            y_trues.extend([y.item() for y in y_true])
            probs.extend([torch.sigmoid(p).item() for p in logits])

        y_preds = [int(p > cut_point) for p in probs]
        roc_auc = roc_auc_score(y_true=y_trues, y_score=y_preds)
        avg_loss = losses / len(dataloader.sampler)
        return {"avg_loss": avg_loss, "roc_auc": roc_auc, "probs": probs, "y_preds": y_preds}


def evaluate_graph(model, dataloader, criterion, device):
    y_trues = []
    probs = []
    losses = 0.0

    model.eval()
    with torch.no_grad():
        for graph, label in dataloader:
            graph = graph.to(device)
            y_true = label.unsqueeze(1).float().to(device)

            logits = model(graph)
            loss = criterion(logits, y_true)

            losses += (loss.item() * y_true.size(0))
            y_trues.extend([y.item() for y in y_true])
            probs.extend([torch.sigmoid(p).item() for p in logits])

        y_preds = [int(p > 0.5) for p in probs]
        precision = precision_score(y_true=y_trues, y_pred=y_preds)
        avg_loss = losses / len(dataloader.sampler)
        return {"avg_loss": avg_loss, "precision": precision, "probs": probs, "y_preds": y_preds}
