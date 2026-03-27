import os

import torch
from tqdm import tqdm

from training.metrics import compute_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in tqdm(dataloader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()

        # Gradient clipping — prevents explosive gradients in transformer layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(output, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_targets, all_preds)

    return avg_loss, metrics


def train(
    model,
    train_dl,
    val_dl,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_path,
    patience=15
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_mcc": []
    }
    best_mcc = -1.0
    wait = 0

    for epoch in range(1, num_epochs + 1):
        epoch_train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        epoch_val_loss, metrics = evaluate(model, val_dl, criterion, device)

        # Step CosineAnnealingWarmRestarts (or any other scheduler) every epoch
        scheduler.step()

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(metrics['accuracy'])
        history["val_mcc"].append(metrics['mcc'])

        print(
            f"epoch {epoch}/{num_epochs}   "
            f"train={epoch_train_loss:.4f}   "
            f"val={epoch_val_loss:.4f}   "
            f"acc={metrics['accuracy']:.4f}   "
            f"mcc={metrics['mcc']:.4f}"
        )

        if metrics['mcc'] > best_mcc:
            best_mcc = metrics['mcc']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return history
