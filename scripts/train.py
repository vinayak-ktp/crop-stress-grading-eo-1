import argparse
import json
import os

import torch
import torch.nn as nn

from data_pipeline.dataloader import get_class_weights, get_loader
from models.ensemble import build_ensemble
from models.hybrid import HybridModel
from models.inception_1d import InceptionNet1D
from models.resnet_1d import ResNet1D
from models.transformer import TransformerModel
from training.metrics import compute_metrics, get_predictions
from training.trainer import train

MODEL_MAP = {
    "resnet": ResNet1D,
    "transformer": TransformerModel,
    "hybrid": HybridModel,
    "inception": InceptionNet1D,
}

MODEL_COLORS = {
    "resnet": "#4878CF",
    "transformer": "#D65F5F",
    "hybrid": "#6ACC65",
    "inception": "#B47CC7",
    "ensemble": "#FFA500",
}


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True,
                    choices=["resnet", "transformer", "hybrid", "inception", "all"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--mixup", action="store_true", help="Enable mixup augmentation")
parser.add_argument("--ensemble", action="store_true",
                    help="Run soft-voting ensemble after individual training")
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

splits_dir = os.path.join("data", "splits")
train_loader = get_loader(
    os.path.join(splits_dir, "train.csv"),
    batch_size=args.batch_size, shuffle=True,
    augment=True, weighted_sampling=True
)
val_loader = get_loader(os.path.join(splits_dir, "val.csv"), batch_size=args.batch_size)
test_loader = get_loader(os.path.join(splits_dir, "test.csv"), batch_size=args.batch_size)

input_dim = train_loader.dataset.X.shape[1]
num_classes = len(train_loader.dataset.y.unique())

# Compute class weights for balanced loss
class_weights = get_class_weights(train_loader.dataset).to(device)
print(f"Class weights: {class_weights.tolist()}")

models_to_train = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

for model_name in models_to_train:
    print(f"\nTraining {model_name.upper()}")

    model = MODEL_MAP[model_name](input_dim, num_classes).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = os.path.join("checkpoints", f"{model_name}.pt")

    history = train(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, device, args.epochs, save_path, patience=20,
        use_mixup=args.mixup, mixup_alpha=0.4, max_grad_norm=1.0,
        scheduler_type='cosine'
    )

    test_preds, test_targets = get_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_targets, test_preds)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test MCC: {test_metrics['mcc']:.4f}")

    results_dir = os.path.join("_results", "raw")
    os.makedirs(results_dir, exist_ok=True)

    out_data = {
        "history": history,
        "test_metrics": test_metrics
    }

    with open(os.path.join(results_dir, f"{model_name}.json"), "w") as f:
        json.dump(out_data, f)

if args.ensemble:
    print("Soft-voting ensemble")
    checkpoint_paths = {}
    for name in MODEL_MAP:
        cp = os.path.join("checkpoints", f"{name}.pt")
        if os.path.exists(cp):
            checkpoint_paths[name] = cp

    if len(checkpoint_paths) >= 2:
        ensemble = build_ensemble(MODEL_MAP, input_dim, num_classes, checkpoint_paths, device)
        ensemble.eval()

        # Evaluate ensemble - raw logit-like output (already softmaxed in forward)
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                probs = ensemble(X_batch)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())

        import numpy as np
        test_metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        print(f"Ensemble Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Ensemble Test MCC: {test_metrics['mcc']:.4f}")

        results_dir = os.path.join("_results", "raw")
        with open(os.path.join(results_dir, "ensemble.json"), "w") as f:
            json.dump({"history": {}, "test_metrics": test_metrics}, f)
    else:
        print(f"Need at least 2 trained models for ensemble, found {len(checkpoint_paths)}")
