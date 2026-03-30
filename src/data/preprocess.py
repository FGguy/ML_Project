"""
Audio preprocessing: resample and mono conversion for all flattened clips.

Reads each .wav file from data/raw_flat/, resamples from source_sr to
target_sr, converts to mono by averaging channels, and overwrites the file
in-place. Adds sample_rate and channels columns to flat_index.csv to record
the result.

Run this after flatten.py and before distortion.py.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample

from src.utils import load_config


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Convert a multi-channel waveform to mono by averaging all channels.

    Args:
        waveform (torch.Tensor): Audio tensor of shape (channels, samples).

    Returns:
        torch.Tensor: Mono tensor of shape (1, samples).
    """
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def resample_waveform(
    waveform: torch.Tensor, source_sr: int, target_sr: int
) -> torch.Tensor:
    """
    Resample a waveform tensor from source_sr to target_sr.

    Uses torchaudio.transforms.Resample. If source_sr == target_sr the
    waveform is returned unchanged.

    Args:
        waveform (torch.Tensor): Audio tensor of shape (channels, samples).
        source_sr (int): Original sample rate in Hz.
        target_sr (int): Desired sample rate in Hz.

    Returns:
        torch.Tensor: Resampled tensor of shape (channels, new_samples).
    """
    if source_sr == target_sr:
        return waveform
    resampler = Resample(orig_freq=source_sr, new_freq=target_sr)
    return resampler(waveform)


def preprocess_file(file_path: Path, source_sr: int, target_sr: int) -> None:
    """
    Load, resample, convert to mono, and overwrite a single .wav file.

    Args:
        file_path (Path): Path to the .wav file to process (modified in-place).
        source_sr (int): Expected source sample rate in Hz.
        target_sr (int): Target sample rate in Hz.

    Returns:
        None
    """
    waveform, sr = torchaudio.load(str(file_path))
    # Use actual sr from file rather than assuming source_sr
    waveform = resample_waveform(waveform, sr, target_sr)
    waveform = to_mono(waveform)
    torchaudio.save(str(file_path), waveform, target_sr)


def preprocess_dataset(config_path: str = "configs/base_config.yaml") -> None:
    """
    Preprocess all files listed in flat_index.csv: resample and mono conversion.

    Overwrites each file in data/raw_flat/ in-place. Appends sample_rate and
    channels columns to flat_index.csv to record the post-processing state.

    Args:
        config_path (str): Path to the base YAML config file.

    Returns:
        None
    """
    cfg = load_config(config_path)
    flat_dir = Path(cfg["paths"]["raw_flat_dir"])
    index_path = Path(cfg["paths"]["flat_index"])
    source_sr = cfg["audio"]["source_sr"]
    target_sr = cfg["audio"]["target_sr"]

    df = pd.read_csv(index_path)

    # Skip if already preprocessed (column present and filled)
    already_done = "sample_rate" in df.columns and df["sample_rate"].notna().all()
    if already_done:
        print("Preprocessing already applied — skipping.")
        return

    print(f"Preprocessing {len(df)} files: {source_sr} Hz -> {target_sr} Hz, mono...")

    for i, row in df.iterrows():
        file_path = flat_dir / row["flat_filename"]
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file not found: {file_path}")
        preprocess_file(file_path, source_sr, target_sr)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} done")

    df["sample_rate"] = target_sr
    df["channels"] = 1
    df.to_csv(index_path, index=False)
    print(f"Done. Updated {index_path} with sample_rate and channels columns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample and convert IDMT-SMT flat clips to mono."
    )
    parser.add_argument(
        "--config",
        default="configs/base_config.yaml",
        help="Path to the base YAML config file.",
    )
    args = parser.parse_args()
    preprocess_dataset(args.config)
