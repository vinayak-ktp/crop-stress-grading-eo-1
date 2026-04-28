import numpy as np
import torch
import torch.nn as nn

activations = {}
gradients = {}

def fwd_hook(*args):
    activations['a'] = args[2].detach()

def bwd_hook(*args):
    gradients['g'] = args[2][0].detach()


def _find_last_conv1d(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            last = m
    return last


def compute_gradcam(model, x, target_class, device):
    activations.clear()
    gradients.clear()
    model.eval()
    layer = _find_last_conv1d(model)
    if layer is None:
        return None

    fwd_h = layer.register_forward_hook(fwd_hook)
    bwd_h = layer.register_full_backward_hook(bwd_hook)

    x = x.to(device).requires_grad_(True)
    out = model(x)
    model.zero_grad()
    out[0, target_class].backward()

    fwd_h.remove()
    bwd_h.remove()

    alpha = gradients['g'].mean(dim=-1, keepdim=True)
    cam = (alpha * activations['a']).sum(dim=1).squeeze(0)
    cam = torch.relu(cam).cpu().numpy()

    seq_len = x.shape[1]
    cam_up = np.interp(np.linspace(0, len(cam) - 1, seq_len), np.arange(len(cam)), cam)
    if cam_up.max() > 0:
        cam_up /= cam_up.max()
    return cam_up


def gradcam_all_classes(model, loader, num_classes, device, n_samples=64):
    model.eval()
    per_class = {c: [] for c in range(num_classes)}
    max_per_class = max(n_samples // num_classes, 1)
    for x, y in loader:
        if all(len(v) >= max_per_class for v in per_class.values()):
            break
        for i in range(x.shape[0]):
            c = y[i].item()
            if len(per_class[c]) >= max_per_class:
                continue
            xi = x[i:i+1].to(device)
            cam = compute_gradcam(model, xi, c, device)
            if cam is not None:
                per_class[c].append(cam)
    return {c: np.stack(v).mean(axis=0) if v else None for c, v in per_class.items()}
