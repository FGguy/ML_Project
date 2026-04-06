"""
Shared training loop for all three models.

Usage:
    python -m src.train --model rnn
    python -m src.train --model crdnn_audio
    python -m src.train --model crdnn

Each model reads its own YAML config for hyperparameters. The best checkpoint
(lowest val loss) is saved to checkpoints/{model}_best.pt. Training stops
early if val loss does not improve for `patience` epochs.

Dataset routing (enforced — do not mix):
    rnn, crdnn_audio  →  RawAudioDataset
    crdnn             →  SpectrogramDataset
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import RawAudioDataset, SpectrogramDataset, collate_variable_length
from src.models.crdnn_raw_audio import build_audio_crdnn
from src.models.crdnn_spectrogram import build_crdnn
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
        model_name (str): One of 'rnn', 'crdnn_audio', 'crdnn'.
        cfg (dict): Model config dictionary.

    Returns:
        nn.Module: Instantiated model.

    Raises:
        ValueError: If model_name is not recognised.
    """
    if model_name == "rnn":
        return build_rnn(cfg)
    if model_name == "crdnn_audio":
        return build_audio_crdnn(cfg)
    if model_name == "crdnn":
        return build_crdnn(cfg)
    raise ValueError(f"Unknown model: {model_name!r}")


def get_datasets(model_name: str, base_cfg: dict, model_cfg: dict = None):
    """
    Return train and val Dataset instances for the given model.

    RNN and 1D CNN use raw audio; CRDNN uses pre-computed mel spectrograms.
    For CRDNN, SpecAugment is applied to the training split when spec_augment
    params are present in model_cfg.

    Args:
        model_name (str): One of 'rnn', 'crdnn_audio', 'crdnn'.
        base_cfg (dict): Base config containing split paths.
        model_cfg (dict | None): Model config, used for SpecAugment params.

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    splits_dir = base_cfg["paths"]["splits_dir"]
    train_csv = f"{splits_dir}/train.csv"
    val_csv = f"{splits_dir}/val.csv"

    if model_name == "crdnn":
        spec_aug_cfg = (model_cfg or {}).get("spec_augment", None)
        return (
            SpectrogramDataset(train_csv, train=True, spec_augment_cfg=spec_aug_cfg),
            SpectrogramDataset(val_csv, train=False),
        )
    return RawAudioDataset(train_csv), RawAudioDataset(val_csv)


def compute_class_weights(
    train_csv: str, num_classes: int, device: torch.device
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training CSV.

    Args:
        train_csv (str): Path to train.csv.
        num_classes (int): Total number of classes.
        device (torch.device): Device to move the weight tensor to.

    Returns:
        torch.Tensor: Shape (num_classes,) weight tensor for CrossEntropyLoss.
    """
    labels = pd.read_csv(train_csv)["label"].tolist()
    counts = Counter(labels)
    # Inverse frequency; normalised so weights sum to num_classes
    weights = torch.tensor(
        [1.0 / counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32
    )
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple:
    """
    Apply Mixup to a training batch.

    Linearly blends pairs of examples and their labels so the model learns
    smoother decision boundaries — useful for spectrally close pedal types.

    Ref: Purwins et al. (1905.00078)

    Args:
        x (torch.Tensor): Input batch of any shape (batch, ...).
        y (torch.Tensor): Integer class labels of shape (batch,).
        alpha (float): Beta distribution concentration parameter.

    Returns:
        tuple: (x_mix, y_a, y_b, lam) where lam is the mixing coefficient.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam


def run_epoch(model, loader, criterion, optimizer, device, train: bool, mixup_alpha: float = 0.0):
    """
    Run one pass over a DataLoader, returning avg loss and accuracy.

    Handles both fixed-length batches (2-tuple) and variable-length batches
    (3-tuple with lengths). Lengths are passed to models that support packed
    sequences (RNN baseline, AudioCRDNN).

    When train=True and mixup_alpha > 0, Mixup is applied to each batch.

    Args:
        model (nn.Module): The model to run.
        loader (DataLoader): DataLoader for this split.
        criterion: Loss function.
        optimizer: Optimizer (ignored when train=False).
        device (torch.device): Device to run on.
        train (bool): If True, backprop and update weights.
        mixup_alpha (float): Mixup Beta parameter. 0.0 disables Mixup.

    Returns:
        tuple[float, float]: (avg_loss, accuracy)
    """
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            if len(batch) == 3:
                x, y, lengths = batch
            else:
                x, y = batch
                lengths = None

            x, y = x.to(device), y.to(device)

            if train and mixup_alpha > 0:
                x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
                logits = model(x, lengths=lengths) if lengths is not None else model(x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                # Accuracy uses the dominant label for logging purposes
                preds = logits.argmax(dim=1)
                correct += (lam * (preds == y_a).float() + (1 - lam) * (preds == y_b).float()).sum().item()
            else:
                logits = model(x, lengths=lengths) if lengths is not None else model(x)
                loss = criterion(logits, y)
                correct += (logits.argmax(dim=1) == y).sum().item()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            total += len(y)

    return total_loss / total, correct / total


def train_model(model_name: str) -> None:
    """
    Full training loop for the specified model.

    Loads config, builds datasets and model, trains with early stopping and
    cosine LR annealing, and saves the best checkpoint to
    checkpoints/{model_name}_best.pt.

    Args:
        model_name (str): One of 'rnn', 'crdnn_audio', 'crdnn'.
    """
    base_cfg = load_config("configs/base_config.yaml")
    model_cfg = load_config(
        "configs/rnn_baseline_config.yaml"
        if model_name == "rnn"
        else f"configs/{model_name}_config.yaml"
    )

    seed = 42
    set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} on {device}")

    train_ds, val_ds = get_datasets(model_name, base_cfg, model_cfg)
    collate_fn = collate_variable_length if model_name in ("rnn", "crdnn_audio") else None
    train_loader = DataLoader(
        train_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=model_cfg["num_workers"],
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=model_cfg["num_workers"],
        collate_fn=collate_fn,
    )

    model = get_model(model_name, model_cfg).to(device)

    # Class-weighted loss to handle any imbalance from the grouped split.
    # Ref: Levy et al. (s13634-022-00933-9)
    if model_cfg.get("use_class_weights", False):
        splits_dir = base_cfg["paths"]["splits_dir"]
        weights = compute_class_weights(
            f"{splits_dir}/train.csv", model_cfg["num_classes"], device
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_cfg["lr"],
        weight_decay=model_cfg.get("weight_decay", 0.0),
    )

    # Cosine annealing decays LR smoothly to ~0 over the training run.
    # Ref: Purwins et al. (1905.00078), Rossi et al. (DAFx25_paper_16)
    scheduler = None
    if model_cfg.get("scheduler", "none") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=model_cfg["epochs"]
        )

    mixup_alpha = model_cfg.get("mixup_alpha", 0.0)

    best_val_loss = float("inf")
    patience_counter = 0
    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = f"checkpoints/{model_name}_best.pt"

    for epoch in range(1, model_cfg["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, mixup_alpha=mixup_alpha,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False,
        )

        if scheduler is not None:
            scheduler.step()

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
        choices=["rnn", "crdnn_audio", "crdnn"],
        required=True,
        help="Model to train.",
    )
    args = parser.parse_args()
    train_model(args.model)
