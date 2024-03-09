import torch
from sklearn.metrics import roc_auc_score, precision_score


def train(model, dataloader, optimizer, criterion, device, cut_point: float):
    y_trues = []
    losses = 0.0
    probs = []

    model.train()

    for batch in dataloader:
        x_emb = batch["x_emb"].to(device)
        x_cont = batch["x_cont"].to(device)
        y_true = batch["y"].float().to(device)

        optimizer.zero_grad()
        logits = model(
            x_cont=x_cont,
            x_emb=x_emb
        )
        loss = criterion(logits.squeeze(1), y_true)
        loss.backward()
        optimizer.step()

        losses += (loss.item() * y_true.size(0))
        y_trues.extend([y.item() for y in y_true])
        probs.extend([torch.sigmoid(p).item() for p in logits])

    y_preds = [int(p > cut_point) for p in probs]
    roc_auc = roc_auc_score(y_true=y_trues, y_score=y_preds)
    avg_loss = losses / len(dataloader.sampler)
    return {"avg_loss": avg_loss, "roc_auc": roc_auc, "probs": probs, "y_preds": y_preds}


def train_graph(model, dataloader, optimizer, criterion, device):
    model.train()
    y_trues = []
    probs = []
    losses = 0.0

    for graph, label in dataloader:
        graph = graph.to(device)
        y_true = label.unsqueeze(1).float().to(device)

        optimizer.zero_grad()
        logits = model(graph)

        loss = criterion(logits, y_true)

        loss.backward()
        optimizer.step()

        losses += (loss.item() * y_true.size(0))
        y_trues.extend([y.item() for y in y_true])
        probs.extend([torch.sigmoid(p).item() for p in logits])

    y_preds = [int(p > 0.5) for p in probs]
    precision = precision_score(y_true=y_trues, y_pred=y_preds)
    avg_loss = losses / len(dataloader.sampler)
    return {"avg_loss": avg_loss, "precision": precision, "probs": probs, "y_preds": y_preds}
