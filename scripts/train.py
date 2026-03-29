import argparse
import json
import os

import torch
import torch.nn as nn

from data_pipeline.dataloader import get_loader
from models.hybrid import HybridModel
from models.paper_cnn import PaperCNN1D
from models.resnet import ResNet1D
from models.transformer import TransformerModel
from training.metrics import compute_metrics, get_predictions
from training.trainer import train

MODEL_MAP = {
    "paper_cnn": PaperCNN1D,
    "resnet": ResNet1D,
    "transformer": TransformerModel,
    "hybrid": HybridModel,
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=["resnet", "transformer", "hybrid", "paper_cnn", "all"])
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--exp", type=str, required=True, help="Experiment name used to namespace outputs")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--label_smoothing", type=float, default=0.1)
parser.add_argument("--patience", type=int, default=15)
parser.add_argument("--T0", type=int, default=10, help="CosineAnnealingWarmRestarts T_0")
parser.add_argument("--T_mult", type=int, default=2, help="CosineAnnealingWarmRestarts T_mult")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

splits_dir = os.path.join("data", "splits")
train_loader = get_loader(os.path.join(splits_dir, "train.csv"), batch_size=args.batch_size, shuffle=True)
val_loader = get_loader(os.path.join(splits_dir, "val.csv"), batch_size=args.batch_size)
test_loader = get_loader(os.path.join(splits_dir, "test.csv"), batch_size=args.batch_size)

input_dim = train_loader.dataset.X.shape[1]
num_classes = len(train_loader.dataset.y.unique())

y_train = train_loader.dataset.y
class_counts = torch.bincount(y_train, minlength=num_classes).float()
class_weights = (1.0 / class_counts)
class_weights = (class_weights / class_weights.sum() * num_classes).to(device)
print(f"Class weights: {class_weights.tolist()}")

models_to_train = list(MODEL_MAP.keys()) if args.model == "all" else [args.model]

results_dir = os.path.join("_results", "raw", args.exp)
checkpoints_dir = os.path.join("checkpoints", args.exp)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

hparams = {
    "exp": args.exp,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "label_smoothing": args.label_smoothing,
    "patience": args.patience,
    "scheduler": "CosineAnnealingWarmRestarts",
    "T0": args.T0,
    "T_mult": args.T_mult,
    "input_dim": input_dim,
    "num_classes": num_classes,
    "class_weights": class_weights.tolist(),
}

for model_name in models_to_train:
    print(f"\nTraining {model_name.upper()}")

    model = MODEL_MAP[model_name](input_dim, num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.T0, T_mult=args.T_mult, eta_min=1e-6
    )

    save_path = os.path.join(checkpoints_dir, f"{model_name}.pt")

    history = train(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, device, args.epochs, save_path, patience=args.patience
    )

    test_preds, test_targets = get_predictions(model, test_loader, device)
    test_metrics = compute_metrics(test_targets, test_preds)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test MCC: {test_metrics['mcc']:.4f}")

    out_data = {
        "hparams": {**hparams, "model": model_name, "total_params": total_params},
        "history": history,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(results_dir, f"{model_name}.json"), "w") as f:
        json.dump(out_data, f)
