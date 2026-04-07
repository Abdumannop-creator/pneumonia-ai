import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(pneumonia)
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(targets, probs)
    f1 = f1_score(targets, preds)
    recall = recall_score(targets, preds)

    return {
        "auc": float(auc),
        "f1": float(f1),
        "recall": float(recall),
    }