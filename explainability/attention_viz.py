from functools import partial

import numpy as np
import torch

_stored_attentions = []

def _patched_forward(layer, orig_forward, src, src_mask=None, src_key_padding_mask=None, **kwargs):
    _, attn_w = layer.self_attn(
        src, src, src,
        attn_mask=src_mask,
        key_padding_mask=src_key_padding_mask,
        need_weights=True,
        average_attn_weights=True,
    )
    _stored_attentions.append(attn_w.detach().cpu())
    return orig_forward(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)


def _get_encoder(model):
    if hasattr(model, 'transformer_encoder'):
        return model.transformer_encoder
    if hasattr(model, 'transformer'):
        return model.transformer
    return None


def _patch_encoder_layers(encoder):
    _stored_attentions.clear()
    for layer in encoder.layers:
        layer.forward = partial(_patched_forward, layer, layer.forward)
    return _stored_attentions


def attention_all_classes(model, loader, num_classes, device, n_samples=64):
    encoder = _get_encoder(model)
    if encoder is None:
        return None
    model.eval()
    storage = _patch_encoder_layers(encoder)
    per_class = {c: [] for c in range(num_classes)}
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            storage.clear()
            _ = model(x)
            if not storage:
                break
            attn = torch.stack(storage, dim=0).mean(dim=0)
            diag = torch.diagonal(attn, dim1=-2, dim2=-1).cpu().numpy()
            for i in range(x.shape[0]):
                per_class[y[i].item()].append(diag[i])
                count += 1
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
    return {c: np.stack(v).mean(axis=0) if v else None for c, v in per_class.items()}
