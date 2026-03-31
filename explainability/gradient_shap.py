import numpy as np
import torch
from captum.attr import GradientShap


def gradientshap_all_classes(model, loader, num_classes, device, n_samples=64):
    model.eval()
    explainer = GradientShap(model)
    samples, labels = [], []
    count = 0
    for x, y in loader:
        samples.append(x)
        labels.append(y)
        count += x.shape[0]
        if count >= n_samples:
            break
    X = torch.cat(samples, dim=0)[:n_samples].to(device)
    Y = torch.cat(labels, dim=0)[:n_samples]
    baseline = torch.zeros_like(X).to(device)
    attrs = explainer.attribute(X, baseline, target=Y.to(device)).detach().cpu().numpy()
    attrs = np.abs(attrs.squeeze(-1))
    per_class = {c: [] for c in range(num_classes)}
    for i in range(len(Y)):
        per_class[Y[i].item()].append(attrs[i])
    return {c: np.stack(v).mean(axis=0) if v else np.zeros(attrs.shape[1]) for c, v in per_class.items()}
