"""
Amplitude normalization for all processed audio clips.

Applies per-clip peak normalization to every file listed in the split CSVs:
    waveform = waveform / waveform.abs().max()

This is a per-clip operation with no statistics shared across clips, so it
carries no risk of data leakage between splits. Files are overwritten in-place
in data/processed/audio/.

Run this after split.py and before augment.py.
"""

import argparse
from pathlib import Path

import pandas as pd
import torchaudio

from src.utils import load_config


def peak_normalize(waveform):
    """
    Normalize a waveform tensor to the range [-1, 1] by its absolute peak.

    Args:
        waveform (torch.Tensor): Audio tensor of shape (channels, samples).

    Returns:
        torch.Tensor: Peak-normalized tensor of the same shape.
    """
    peak = waveform.abs().max()
    if peak > 0:
        return waveform / peak
    return waveform


def normalize_file(file_path: Path, sample_rate: int) -> None:
    """
    Load a .wav file, peak-normalize it, and overwrite it in-place.

    Args:
        file_path (Path): Path to the .wav file to normalize.
        sample_rate (int): Sample rate to use when saving the file.

    Returns:
        None
    """
    waveform, _ = torchaudio.load(str(file_path))
    waveform = peak_normalize(waveform)
    torchaudio.save(str(file_path), waveform, sample_rate)


def normalize_dataset(config_path: str = "configs/base_config.yaml") -> None:
    """
    Peak-normalize every clip listed across all split CSVs.

    Collects unique file paths from train.csv, val.csv, and test.csv and
    normalizes each one in-place. Skips any file that does not exist on disk.

    Args:
        config_path (str): Path to the base YAML config file.

    Returns:
        None
    """
    cfg = load_config(config_path)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    target_sr = cfg["audio"]["target_sr"]

    all_paths = set()
    for split_name in ("train", "val", "test"):
        split_csv = splits_dir / f"{split_name}.csv"
        df = pd.read_csv(split_csv)
        all_paths.update(df["filepath"].tolist())

    total = len(all_paths)
    print(f"Normalizing {total} clips...")

    for i, filepath in enumerate(sorted(all_paths)):
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Expected file not found: {path}")
        normalize_file(path, target_sr)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} done")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Peak-normalize all processed audio clips in-place."
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    normalize_dataset(args.config)
