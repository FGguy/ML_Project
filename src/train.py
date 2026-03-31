"""
Shared training loop for all three models.

Usage:
    python -m src.train --model rnn
    python -m src.train --model cnn1d
    python -m src.train --model crdnn

Each model reads its own YAML config for hyperparameters. The best checkpoint
(lowest val loss) is saved to checkpoints/{model}_best.pt. Training stops
early if val loss does not improve for `patience` epochs.

Dataset routing (enforced — do not mix):
    rnn, cnn1d  →  RawAudioDataset
    crdnn       →  SpectrogramDataset
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import RawAudioDataset, SpectrogramDataset
from src.models.cnn1d import build_cnn1d
from src.models.crdnn import build_crdnn
from src.models.rnn_baseline import build_rnn
from src.utils import load_config


def set_seeds(seed: int) -> None:
    """
    Fix all random seeds for reproducibility.

    Args:
        seed (int): Seed value to apply globally.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_model(model_name: str, cfg: dict) -> nn.Module:
    """
    Instantiate the requested model from its config dict.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
        cfg (dict): Model config dictionary.

    Returns:
        nn.Module: Instantiated model.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "rnn":
        return build_rnn(cfg)
    if model_name == "cnn1d":
        return build_cnn1d(cfg)
    if model_name == "crdnn":
        return build_crdnn(cfg)
    raise ValueError(f"Unknown model: {model_name!r}")


def get_datasets(model_name: str, base_cfg: dict):
    """
    Return train and val Dataset instances for the given model.

    RNN and 1D CNN use raw audio; CRDNN uses pre-computed mel spectrograms.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
        base_cfg (dict): Base config containing split paths.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    splits_dir = base_cfg["paths"]["splits_dir"]
    train_csv = f"{splits_dir}/train.csv"
    val_csv = f"{splits_dir}/val.csv"

    if model_name == "crdnn":
        return SpectrogramDataset(train_csv), SpectrogramDataset(val_csv)
    return RawAudioDataset(train_csv), RawAudioDataset(val_csv)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    Run one pass over a DataLoader, returning avg loss and accuracy.

    Args:
        model (nn.Module): The model to run.
        loader (DataLoader): DataLoader for this split.
        criterion: Loss function.
        optimizer: Optimizer (ignored when train=False).
        device (torch.device): Device to run on.
        train (bool): If True, backprop and update weights.

    Returns:
        tuple[float, float]: (avg_loss, accuracy)
    """
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total


def train_model(model_name: str) -> None:
    """
    Full training loop for the specified model.

    Loads config, builds datasets and model, trains with early stopping,
    and saves the best checkpoint to checkpoints/{model_name}_best.pt.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
    """
    base_cfg = load_config("configs/base_config.yaml")
    model_cfg = load_config(
        f"configs/{model_name}_config.yaml"
        if model_name != "rnn"
        else "configs/rnn_baseline_config.yaml"
    )

    seed = 42
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}")

    train_ds, val_ds = get_datasets(model_name, base_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=model_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=model_cfg["num_workers"],
    )

    model = get_model(model_name, model_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"])

    best_val_loss = float("inf")
    patience_counter = 0
    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = f"checkpoints/{model_name}_best.pt"

    for epoch in range(1, model_cfg["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= model_cfg["patience"]:
                patience = model_cfg["patience"]
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"Best val loss: {best_val_loss:.4f} — checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a pedal classification model.")
    parser.add_argument(
        "--model",
        choices=["rnn", "cnn1d", "crdnn"],
        required=True,
        help="Model to train.",
    )
    args = parser.parse_args()
    train_model(args.model)
