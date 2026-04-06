"""
Final evaluation of a trained model on the held-out test set.

Usage:
    python -m src.evaluate --model rnn
    python -m src.evaluate --model cnn1d
    python -m src.evaluate --model crdnn

Loads the best checkpoint from checkpoints/{model}_best.pt and reports:
  - Test accuracy
  - Row-normalised confusion matrix (saved to results/{model}_confusion.png)
  - Per-class precision, recall and F1 (sklearn classification_report)
  - All numeric results saved to results/{model}_results.json

Every output is framed around the research question: does time-domain (raw
audio) or frequency-domain (mel spectrogram) better capture the non-linear
clipping characteristics of distortion pedals?

The test set is touched exactly once — here. Never call this script during
hyperparameter tuning.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import RawAudioDataset, SpectrogramDataset
from src.models.crdnn_raw_audio import build_cnn1d
from src.models.crdnn_spectrogram import build_crdnn
from src.models.rnn_baseline import build_rnn
from src.utils import load_config

DOMAIN_LABEL = {
    "rnn": "time domain (raw audio)",
    "cnn1d": "time domain (raw audio)",
    "crdnn": "frequency domain (mel spectrogram)",
}


def get_model(model_name: str, cfg: dict) -> torch.nn.Module:
    """
    Instantiate the requested model from its config dict.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
        cfg (dict): Model config dictionary.

    Returns:
        torch.nn.Module: Instantiated model.

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


def get_test_dataset(model_name: str, base_cfg: dict):
    """
    Return the test Dataset for the given model type.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
        base_cfg (dict): Base config containing split paths.

    Returns:
        Dataset: Test split dataset.
    """
    test_csv = f"{base_cfg['paths']['splits_dir']}/test.csv"
    if model_name == "crdnn":
        return SpectrogramDataset(test_csv)
    return RawAudioDataset(test_csv)


def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over the test loader and collect predictions and labels.

    Args:
        model (torch.nn.Module): Trained model in eval mode.
        loader (DataLoader): Test DataLoader.
        device (torch.device): Device to run on.

    Returns:
        tuple[np.ndarray, np.ndarray]: (all_preds, all_labels)
    """
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list, model_name: str, out_path: Path
) -> None:
    """
    Plot and save a row-normalised confusion matrix heatmap.

    Rows represent true labels, columns represent predicted labels. Each cell
    shows recall for that true class, making it easy to spot which pedals are
    most often confused with one another.

    Args:
        cm (np.ndarray): Confusion matrix of shape (n_classes, n_classes).
        class_names (list): Ordered list of class name strings.
        model_name (str): Model name used in the plot title.
        out_path (Path): File path to save the figure.
    """
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    domain = DOMAIN_LABEL[model_name]
    ax.set_title(f"{model_name.upper()} — {domain}\nRow-normalised confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def evaluate(model_name: str) -> None:
    """
    Load the best checkpoint and run full evaluation on the test set.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
    """
    base_cfg = load_config("configs/base_config.yaml")
    cfg_file = (
        "configs/rnn_baseline_config.yaml"
        if model_name == "rnn"
        else f"configs/{model_name}_config.yaml"
    )
    model_cfg = load_config(cfg_file)

    checkpoint_path = Path(f"checkpoints/{model_name}_best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run train first."
        )

    Path("results").mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model and load weights
    model = get_model(model_name, model_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Warm up lazy modules (CRDNN builds GRU on first forward pass)
    test_ds = get_test_dataset(model_name, base_cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=model_cfg["batch_size"],
        shuffle=False,
        num_workers=model_cfg["num_workers"],
    )

    # Run one dummy batch so lazy submodules are initialised before loading state
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        model(x.to(device))

    model.load_state_dict(checkpoint["model_state"])

    all_preds, all_labels = run_inference(model, test_loader, device)

    # Class names from distortion config, ordered by label index
    dist_cfg = load_config("configs/distortion_config.yaml")
    label_map = dist_cfg["labels"]
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    accuracy = (all_preds == all_labels).mean()
    domain = DOMAIN_LABEL[model_name]

    print(f"\n{'=' * 60}")
    print(f"Model : {model_name.upper()}  ({domain})")
    print(f"Epoch of best checkpoint: {checkpoint['epoch']}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(
        cm, class_names, model_name, Path(f"results/{model_name}_confusion.png")
    )

    results = {
        "model": model_name,
        "domain": domain,
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "test_accuracy": float(accuracy),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
    }
    results_path = Path(f"results/{model_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set."
    )
    parser.add_argument(
        "--model",
        choices=["rnn", "cnn1d", "crdnn"],
        required=True,
        help="Model to evaluate.",
    )
    args = parser.parse_args()
    evaluate(args.model)
