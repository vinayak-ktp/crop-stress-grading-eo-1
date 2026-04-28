import argparse
import os
import tempfile

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_pipeline.dataloader import get_loader
from explainability.attention_viz import attention_all_classes
from explainability.grad_cam import gradcam_all_classes
from explainability.gradient_shap import gradientshap_all_classes
from models.hybrid import HybridModel
from models.paper_cnn import PaperCNN1D
from models.resnet import ResNet1D
from models.transformer import TransformerModel

MODEL_MAP = {
    "paper_cnn": PaperCNN1D,
    "resnet": ResNet1D,
    "transformer": TransformerModel,
    "hybrid": HybridModel,
}

GRADCAM_MODELS = {"paper_cnn", "resnet", "hybrid"}
ATTENTION_MODELS = {"transformer", "hybrid"}


def _infer_input_dim(sd, model_name):
    if model_name == "paper_cnn":
        return sd["classifier.1.weight"].shape[1] // 32 * 2
    if model_name == "transformer":
        return sd["classifier.0.weight"].shape[1] // 64
    if model_name == "hybrid":
        return sd["classifier.1.weight"].shape[1] // 64 * 2
    return None  # resnet: AdaptiveAvgPool1d makes input_dim irrelevant to weights


def _infer_num_classes(sd, model_name):
    if model_name == "paper_cnn":
        return sd["classifier.1.weight"].shape[0]
    if model_name == "transformer":
        return sd["classifier.5.weight"].shape[0]
    if model_name == "hybrid":
        return sd["classifier.6.weight"].shape[0]
    return sd["head.1.weight"].shape[0]


def _save_line(data, out_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    for cls, vals in sorted(data.items()):
        if vals is not None and len(vals) > 0:
            ax.plot(vals, label=f"Class {cls}", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_heatmap(data, out_path, title):
    valid = {c: v for c, v in data.items() if v is not None and len(v) > 0}
    if not valid:
        return
    classes = sorted(valid.keys())
    matrix = np.stack([valid[c] for c in classes])
    fig, ax = plt.subplots(figsize=(12, len(classes) * 0.8 + 1))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"Class {c}" for c in classes])
    ax.set_xlabel("Spectral Band")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_top_bands(data, out_path, k=20):
    valid = [v for v in data.values() if v is not None and len(v) > 0]
    if not valid:
        return
    mean_importance = np.stack(valid).mean(axis=0)
    k = min(k, len(mean_importance))
    top_k = np.argsort(mean_importance)[-k:][::-1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(k), mean_importance[top_k])
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"F_{i}" for i in top_k], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel("Mean Importance")
    ax.set_title(f"Top {k} Spectral Bands by Mean Attribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_explain(model_name, exp, methods, n_samples, out_dir, device):
    ckpt_path = os.path.join("checkpoints", exp, f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    sd = torch.load(ckpt_path, map_location=device)
    input_dim = _infer_input_dim(sd, model_name)
    num_classes = _infer_num_classes(sd, model_name)

    model = MODEL_MAP[model_name](input_dim or 1, num_classes).to(device)
    model.load_state_dict(sd)
    model.eval()

    test_csv = os.path.join("data", "splits", "test.csv")
    df = pd.read_csv(test_csv)
    feature_cols = [c for c in df.columns if c != 'target']
    if input_dim is not None and len(feature_cols) != input_dim:
        df = df[feature_cols[:input_dim] + ['target']]
        tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w')
        df.to_csv(tmp.name, index=False)
        test_loader = get_loader(tmp.name, batch_size=32)
    else:
        test_loader = get_loader(test_csv, batch_size=32)
    os.makedirs(out_dir, exist_ok=True)

    if "gradcam" in methods and model_name in GRADCAM_MODELS:
        result = gradcam_all_classes(model, test_loader, num_classes, device, n_samples)
        _save_line(result, os.path.join(out_dir, "gradcam_mean.png"), f"Grad-CAM ({model_name})", "Saliency")
        _save_heatmap(result, os.path.join(out_dir, "gradcam_heatmap.png"), f"Grad-CAM Heatmap ({model_name})")

    if "gradientshap" in methods:
        result = gradientshap_all_classes(model, test_loader, num_classes, device, n_samples)
        _save_line(result, os.path.join(out_dir, "gradientshap_mean.png"), f"GradientShap ({model_name})", "|Attribution|")
        _save_heatmap(result, os.path.join(out_dir, "gradientshap_heatmap.png"), f"GradientShap Heatmap ({model_name})")
        _save_top_bands(result, os.path.join(out_dir, "top_bands.png"))

    if "attention" in methods and model_name in ATTENTION_MODELS:
        result = attention_all_classes(model, test_loader, num_classes, device, n_samples)
        if result is not None:
            _save_line(result, os.path.join(out_dir, "attention_mean.png"), f"Attention ({model_name})", "Attention Score")
            _save_heatmap(result, os.path.join(out_dir, "attention_heatmap.png"), f"Attention Heatmap ({model_name})")


parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, required=True)
parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()) + ["all"])
parser.add_argument("--method", type=str, default="all", choices=["gradcam", "gradientshap", "attention", "all"])
parser.add_argument("--n_samples", type=int, default=64)
parser.add_argument("--out_dir", type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
methods = ["gradcam", "gradientshap", "attention"] if args.method == "all" else [args.method]
models = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

for model_name in models:
    out = args.out_dir or os.path.join("_results", "explain", args.exp, model_name)
    run_explain(model_name, args.exp, methods, args.n_samples, out, device)
