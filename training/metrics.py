import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)


def get_predictions(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            preds.append(torch.argmax(y_pred, dim=1).cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        'accuracy': acc,
        'f1_score': f1,
        'mcc': mcc,
        'confusion_matrix': cm
    }
