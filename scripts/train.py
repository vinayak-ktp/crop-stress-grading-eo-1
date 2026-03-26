import argparse
import json
import os

import torch
import torch.nn as nn

from data_pipeline.dataloader import get_loader
from models.hybrid import HybridModel
from models.resnet_1d import ResNet1D
from models.transformer import TransformerModel
from training.metrics import compute_metrics, get_predictions
from training.trainer import train

MODEL_MAP = {
    "resnet": ResNet1D,
    "transformer": TransformerModel,
    "hybrid": HybridModel,
}


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["resnet", "transformer", "hybrid", "all"])
parser.add_argument("--epochs", type=int, default=60)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

splits_dir = os.path.join("data", "splits")
train_loader = get_loader(os.path.join(splits_dir, "train.csv"), batch_size=64, shuffle=True)
val_loader   = get_loader(os.path.join(splits_dir, "val.csv"),   batch_size=64)
test_loader  = get_loader(os.path.join(splits_dir, "test.csv"),  batch_size=64)

input_dim   = train_loader.dataset.X.shape[1]
num_classes = len(train_loader.dataset.y.unique())

# ── Compute inverse-frequency class weights to handle imbalance ──────────────
y_train      = train_loader.dataset.y
class_counts = torch.bincount(y_train, minlength=num_classes).float()
class_weights = (1.0 / class_counts)
class_weights = (class_weights / class_weights.sum() * num_classes).to(device)
print(f"Class weights: {class_weights.tolist()}")

models_to_train = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

for model_name in models_to_train:
    print(f"\nTraining {model_name.upper()}")

    model = MODEL_MAP[model_name](input_dim, num_classes).to(device)

    # Weighted loss + label smoothing addresses class imbalance + overconfidence
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # CosineAnnealingWarmRestarts escapes local minima better than ReduceLROnPlateau
    # T_0=10, T_mult=2  →  restarts at epochs 10, 30, 70 …
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    save_path = os.path.join("checkpoints", f"{model_name}.pt")

    history = train(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, device, args.epochs, save_path, patience=15
    )

    test_preds, test_targets = get_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_targets, test_preds)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test MCC:      {test_metrics['mcc']:.4f}")

    results_dir = os.path.join("_results", "raw")
    os.makedirs(results_dir, exist_ok=True)

    out_data = {
        "history": history,
        "test_metrics": test_metrics
    }

    with open(os.path.join(results_dir, f"{model_name}.json"), "w") as f:
        json.dump(out_data, f)
