"""
Random-search hyperparameter tuning for all three models.

Usage:
    python -m src.tune --model rnn     [--trials 10] [--tune-epochs 20]
    python -m src.tune --model cnn1d   [--trials 10] [--tune-epochs 20]
    python -m src.tune --model crdnn   [--trials 10] [--tune-epochs 20]

Each trial samples a hyperparameter combination at random, trains for up to
`tune_epochs` epochs (with a short patience), and records the best val loss.
After all trials the winning parameters are written back into the model's YAML
config so that `src.train` uses them directly.

Only the validation set is used to score trials — the test set is never
touched here.
"""

import argparse
import copy
import random

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.train import get_datasets, get_model, run_epoch, set_seeds
from src.utils import load_config

# Search spaces — lists of values to sample from uniformly at random.
# Only parameters that meaningfully affect convergence are tuned; architecture
# hyperparameters (conv channels, kernel sizes) are kept fixed to avoid
# combinatorial explosion on Colab.
SEARCH_SPACES = {
    "rnn": {
        "lr": [1e-4, 5e-4, 1e-3, 5e-3],
        "hidden_size": [64, 128, 256],
        "batch_size": [16, 32, 64],
    },
    "cnn1d": {
        "lr": [1e-4, 5e-4, 1e-3],
        "dropout": [0.2, 0.3, 0.5],
        "batch_size": [16, 32, 64],
    },
    "crdnn": {
        "lr": [1e-4, 5e-4, 1e-3],
        "gru_hidden": [32, 64, 128],
        "dropout": [0.2, 0.3, 0.5],
        "batch_size": [8, 16],
    },
}


def config_path_for(model_name: str) -> str:
    """
    Return the YAML config file path for a given model name.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.

    Returns:
        str: Path to the model's YAML config file.
    """
    if model_name == "rnn":
        return "configs/rnn_baseline_config.yaml"
    return f"configs/{model_name}_config.yaml"


def sample_params(search_space: dict, rng: random.Random) -> dict:
    """
    Draw one random combination from a search space.

    Args:
        search_space (dict): Maps param name to list of candidate values.
        rng (random.Random): Seeded RNG for reproducible sampling.

    Returns:
        dict: One sampled value per parameter.
    """
    return {key: rng.choice(values) for key, values in search_space.items()}


def run_trial(
    model_name: str,
    base_cfg: dict,
    trial_cfg: dict,
    device: torch.device,
    tune_epochs: int,
    tune_patience: int,
    trial_seed: int,
) -> float:
    """
    Train a model with one hyperparameter configuration and return best val loss.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
        base_cfg (dict): Base pipeline config (paths etc.).
        trial_cfg (dict): Full model config with this trial's sampled params applied.
        device (torch.device): Device to run on.
        tune_epochs (int): Maximum number of epochs per trial.
        tune_patience (int): Early stopping patience within each trial.
        trial_seed (int): Seed for this trial's weight initialisation.

    Returns:
        float: Best validation loss achieved during this trial.
    """
    set_seeds(trial_seed)

    train_ds, val_ds = get_datasets(model_name, base_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=trial_cfg["batch_size"],
        shuffle=True,
        num_workers=trial_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=trial_cfg["batch_size"],
        shuffle=False,
        num_workers=trial_cfg["num_workers"],
    )

    model = get_model(model_name, trial_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_cfg["lr"])

    best_val_loss = float("inf")
    patience_counter = 0

    for _ in range(tune_epochs):
        run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, _ = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tune_patience:
                break

    return best_val_loss


def write_best_params(config_file: str, best_params: dict) -> None:
    """
    Overwrite tuned parameters in the model's YAML config file.

    Only the keys present in best_params are updated; all other config
    values are left unchanged.

    Args:
        config_file (str): Path to the model YAML config file.
        best_params (dict): Param names → best values found by tuning.
    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    cfg.update(best_params)
    with open(config_file, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def tune(model_name: str, n_trials: int, tune_epochs: int) -> None:
    """
    Run random-search hyperparameter tuning and write best params to config.

    Args:
        model_name (str): One of 'rnn', 'cnn1d', 'crdnn'.
        n_trials (int): Number of random configurations to evaluate.
        tune_epochs (int): Maximum epochs to train per trial.
    """
    base_cfg = load_config("configs/base_config.yaml")
    cfg_file = config_path_for(model_name)
    base_model_cfg = load_config(cfg_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_space = SEARCH_SPACES[model_name]
    # Short patience for trials — full patience used in final training
    tune_patience = max(3, base_model_cfg["patience"] // 3)

    rng = random.Random(42)
    best_val_loss = float("inf")
    best_params = {}

    print(f"Tuning {model_name} — {n_trials} trials, up to {tune_epochs} epochs each")
    print(f"Search space: {search_space}\n")

    for trial in range(1, n_trials + 1):
        sampled = sample_params(search_space, rng)
        trial_cfg = copy.deepcopy(base_model_cfg)
        trial_cfg.update(sampled)

        param_str = ", ".join(f"{k}={v}" for k, v in sampled.items())
        print(f"Trial {trial:02d}/{n_trials} | {param_str}", end=" | ", flush=True)

        val_loss = run_trial(
            model_name,
            base_cfg,
            trial_cfg,
            device,
            tune_epochs,
            tune_patience,
            trial_seed=42 + trial,
        )
        print(f"val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = sampled

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Best params:  {best_params}")

    write_best_params(cfg_file, best_params)
    print(f"Written to {cfg_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random-search hyperparameter tuning (val set only)."
    )
    parser.add_argument("--model", choices=["rnn", "cnn1d", "crdnn"], required=True)
    parser.add_argument("--trials", type=int, default=10, help="Number of trials.")
    parser.add_argument(
        "--tune-epochs",
        type=int,
        default=20,
        help="Max epochs per trial.",
    )
    args = parser.parse_args()
    tune(args.model, args.trials, args.tune_epochs)
