import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MODELS = ["paper_cnn", "resnet", "transformer", "hybrid"]

MODEL_COLORS = {
    "paper_cnn": "#E07B39",
    "resnet": "#4878CF",
    "transformer": "#D65F5F",
    "hybrid": "#6ACC65",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _apply_grid_style(ax):
    ax.set_facecolor("#f0f0f0")
    ax.yaxis.grid(True, color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_results(results_dir, model_name):
    json_path = os.path.join(results_dir, f"{model_name}.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def plot_per_model(model_name, data, per_model_dir):
    history = data["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    color = MODEL_COLORS.get(model_name, "#888888")

    os.makedirs(per_model_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], color=color, label="Train Loss", linewidth=1.8)
    ax.plot(epochs, history["val_loss"], color=color, label="Val Loss", linewidth=1.8, linestyle="--", alpha=0.7)
    _apply_grid_style(ax)
    ax.set_title(f"{model_name.upper()} - Training & Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(per_model_dir, f"{model_name}_loss_curve.png"))
    plt.close(fig)

    cm = np.array(data["test_metrics"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                linecolor="white", ax=ax, cbar=False)
    ax.set_title(f"{model_name.upper()} - Test Confusion Matrix")
    ax.set_ylabel("True Class")
    ax.set_xlabel("Predicted Class")
    fig.tight_layout()
    fig.savefig(os.path.join(per_model_dir, f"{model_name}_confusion_matrix.png"))
    plt.close(fig)


def plot_comparison(all_data, comparison_dir, ymin=0, ymax=100):
    os.makedirs(comparison_dir, exist_ok=True)
    model_names = [m for m in MODELS if m in all_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training & Validation Loss Comparison", y=1.01)

    for model_name, data in all_data.items():
        color = MODEL_COLORS.get(model_name, "#888888")
        epochs = range(1, len(data["history"]["train_loss"]) + 1)
        axes[0].plot(epochs, data["history"]["train_loss"], color=color, label=model_name, linewidth=1.8)
        axes[1].plot(epochs, data["history"]["val_loss"], color=color, label=model_name, linewidth=1.8)

    for ax, title in zip(axes, ["Train Loss - All Models", "Validation Loss - All Models"]):
        _apply_grid_style(ax)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(comparison_dir, "loss_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Validation Metrics Comparison", y=1.01)

    for model_name, data in all_data.items():
        color = MODEL_COLORS.get(model_name, "#888888")
        epochs = range(1, len(data["history"]["val_acc"]) + 1)
        axes[0].plot(epochs, data["history"]["val_acc"], color=color, label=model_name, linewidth=1.8)
        axes[1].plot(epochs, data["history"]["val_mcc"], color=color, label=model_name, linewidth=1.8)

    for ax, title, ylabel in zip(
        axes,
        ["Validation Accuracy - All Models", "Validation MCC - All Models"],
        ["Accuracy", "MCC"],
    ):
        _apply_grid_style(ax)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(comparison_dir, "val_metrics_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    metric_keys   = ["accuracy", "f1_score", "mcc"]
    metric_labels = ["Accuracy (%)", "F1 Score (%)", "MCC (×100)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison - Test Set Metrics", y=1.01)

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        colors = [MODEL_COLORS.get(m, "#888888") for m in model_names]
        values = [all_data[m]["test_metrics"][key] * 100 for m in model_names]
        bars = ax.bar(model_names, values, color=colors, width=0.5, zorder=3)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

        _apply_grid_style(ax)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(comparison_dir, "test_metrics_comparison.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Experiment name to load and plot")
    parser.add_argument("--ymin", type=int, default=0, help="Set ylim bottom value")
    parser.add_argument("--ymax", type=int, default=100, help="Set ylim top value")
    args = parser.parse_args()

    results_dir = os.path.join("_results", "raw", args.exp)
    per_model_dir = os.path.join("_results", "plots", args.exp, "per_model")
    comparison_dir = os.path.join("_results", "plots", args.exp, "comparison")

    all_data = {}
    for model in MODELS:
        data = load_results(results_dir, model)
        if data is None:
            print(f"Skipping {model} (not found in {results_dir})")
            continue
        plot_per_model(model, data, per_model_dir)
        all_data[model] = data

    if len(all_data) > 1:
        plot_comparison(all_data, comparison_dir, args.ymin, args.ymax)

    print(f"Plots saved to _results/plots/{args.exp}/")
