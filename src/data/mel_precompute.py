"""
Pre-compute mel spectrograms for all splits and save them as .pt tensors.

Pipeline:
  1. Compute T (fixed time frames) as the median spectrogram length over the
     train set (original clips only — augmented clips share the same length
     distribution). T is written back to base_config.yaml so downstream code
     can read it without recomputing.
  2. Compute per-frequency-bin mean and std from the train set (original clips
     only). Stats are saved to data/splits/mel_stats.pt.
  3. For every clip in all three splits: compute the mel spectrogram, pad or
     crop to T, normalize with train stats, and save to
     data/processed/spectrograms/{stem}.pt.

Normalization is always derived from the train set — val and test clips are
never used to compute statistics.

Run this after augment.py. Dataset classes in dataset.py load these .pt files.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import yaml

from src.utils import load_config


def build_mel_transform(mel_cfg: dict, target_sr: int) -> torch.nn.Sequential:
    """
    Build the mel spectrogram transform pipeline from config parameters.

    Args:
        mel_cfg (dict): Mel config dict with n_fft, hop_length, n_mels,
            f_min, f_max, power keys.
        target_sr (int): Sample rate of the audio files.

    Returns:
        torch.nn.Sequential: Sequential transform applying MelSpectrogram
            followed by AmplitudeToDB.
    """
    mel = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        f_min=mel_cfg["f_min"],
        f_max=mel_cfg["f_max"],
        power=mel_cfg["power"],
    )
    return torch.nn.Sequential(mel, T.AmplitudeToDB())


def pad_or_crop(spec: torch.Tensor, fixed_t: int) -> torch.Tensor:
    """
    Pad or crop the time axis of a spectrogram to a fixed length.

    Crops from the end if too long; zero-pads on the right if too short.

    Args:
        spec (torch.Tensor): Spectrogram of shape (1, n_mels, T).
        fixed_t (int): Target number of time frames.

    Returns:
        torch.Tensor: Spectrogram of shape (1, n_mels, fixed_t).
    """
    t = spec.shape[-1]
    if t >= fixed_t:
        return spec[..., :fixed_t]
    pad_size = fixed_t - t
    return torch.nn.functional.pad(spec, (0, pad_size))


def compute_fixed_t(filepaths: list, transform: torch.nn.Sequential) -> int:
    """
    Compute the median spectrogram length over a list of audio files.

    Args:
        filepaths (list): List of .wav file path strings.
        transform (torch.nn.Sequential): Mel spectrogram transform.

    Returns:
        int: Median number of time frames across all files.
    """
    lengths = []
    for fp in filepaths:
        wav, _ = torchaudio.load(fp)
        spec = transform(wav)
        lengths.append(spec.shape[-1])
    return int(torch.tensor(lengths, dtype=torch.float).median().item())


def compute_train_stats(
    filepaths: list, transform: torch.nn.Sequential, fixed_t: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-frequency-bin mean and std over the training set.

    Each spectrogram is padded/cropped to fixed_t before accumulation so the
    statistics reflect the same input shape seen during training.

    Args:
        filepaths (list): List of training .wav file path strings.
        transform (torch.nn.Sequential): Mel spectrogram transform.
        fixed_t (int): Fixed time dimension to use when accumulating stats.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (mean, std) tensors of shape
            (1, n_mels, 1), ready for broadcasting over the time axis.
    """
    # Accumulate sum and sum-of-squares per frequency bin
    n_mels = None
    total_sum = None
    total_sq_sum = None
    n_frames = 0

    for fp in filepaths:
        wav, _ = torchaudio.load(fp)
        spec = pad_or_crop(transform(wav), fixed_t)  # (1, n_mels, fixed_t)

        if n_mels is None:
            n_mels = spec.shape[1]
            total_sum = torch.zeros(n_mels)
            total_sq_sum = torch.zeros(n_mels)

        # Sum over batch and time dims
        total_sum += spec[0].sum(dim=-1)  # (n_mels,)
        total_sq_sum += spec[0].pow(2).sum(dim=-1)
        n_frames += fixed_t

    mean = (total_sum / n_frames).view(1, n_mels, 1)
    variance = (total_sq_sum / n_frames) - (total_sum / n_frames).pow(2)
    std = variance.clamp(min=1e-8).sqrt().view(1, n_mels, 1)
    return mean, std


def precompute_split(
    filepaths: list,
    stems: list,
    out_dir: Path,
    transform: torch.nn.Sequential,
    fixed_t: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    """
    Compute, normalize, and save mel spectrograms for one split.

    Args:
        filepaths (list): Audio file paths for this split.
        stems (list): Output filename stems (one per filepath).
        out_dir (Path): Directory to write .pt files into.
        transform (torch.nn.Sequential): Mel spectrogram transform.
        fixed_t (int): Fixed time dimension.
        mean (torch.Tensor): Train-set mean of shape (1, n_mels, 1).
        std (torch.Tensor): Train-set std of shape (1, n_mels, 1).

    Returns:
        None
    """
    for fp, stem in zip(filepaths, stems):
        out_path = out_dir / f"{stem}.pt"
        if out_path.exists():
            continue
        wav, _ = torchaudio.load(fp)
        spec = pad_or_crop(transform(wav), fixed_t)
        spec = (spec - mean) / std
        torch.save(spec, out_path)


def mel_precompute(config_path: str = "configs/base_config.yaml") -> None:
    """
    Orchestrate mel spectrogram pre-computation for all three splits.

    Args:
        config_path (str): Path to the base YAML config file.

    Returns:
        None
    """
    cfg = load_config(config_path)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    out_dir = Path(cfg["paths"]["processed_spectrograms_dir"])
    mel_stats_path = Path(cfg["paths"]["mel_stats"])
    target_sr = cfg["audio"]["target_sr"]
    mel_cfg = cfg["mel"]

    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    transform = build_mel_transform(mel_cfg, target_sr)

    # Use original clips only (not _aug) to compute T and stats — aug clips
    # share the same length distribution so this is representative and faster.
    train_originals = train_df[~train_df["filepath"].str.endswith("_aug.wav")]

    fixed_t = mel_cfg.get("fixed_time_frames")
    if fixed_t is None:
        print("Computing fixed_time_frames from train set...")
        fixed_t = compute_fixed_t(train_originals["filepath"].tolist(), transform)
        # Persist T so downstream code doesn't need to recompute it
        with open(config_path) as f:
            raw_cfg = yaml.safe_load(f)
        raw_cfg["mel"]["fixed_time_frames"] = fixed_t
        with open(config_path, "w") as f:
            yaml.dump(raw_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  fixed_time_frames = {fixed_t} (written to {config_path})")

    print("Computing train-set normalization stats...")
    mean, std = compute_train_stats(
        train_originals["filepath"].tolist(), transform, fixed_t
    )
    torch.save({"mean": mean, "std": std}, mel_stats_path)
    print(f"  Saved stats to {mel_stats_path}")

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        filepaths = df["filepath"].tolist()
        stems = [Path(fp).stem for fp in filepaths]
        total = len(filepaths)
        print(f"Processing {split_name} ({total} clips)...")
        precompute_split(filepaths, stems, out_dir, transform, fixed_t, mean, std)
        print("  Done.")

    print(f"\nAll spectrograms saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-compute mel spectrograms for all splits."
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    mel_precompute(args.config)
