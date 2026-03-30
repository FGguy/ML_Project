"""
Data augmentation for the training split only.

For each clip in train.csv one augmented variant is created by randomly
applying one of three techniques:
  - additive white noise at ~30 dB SNR
  - random gain scaling in [0.8, 1.2]
  - cyclic time shift by up to 10% of the clip length

Augmented files are saved alongside the originals in data/processed/audio/
with an ``_aug`` suffix. train.csv is updated in-place to include the new
entries. Val and test splits are never touched.

Run this after normalize.py and before mel_precompute.py.
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torchaudio

from src.utils import load_config


def add_noise(
    waveform: torch.Tensor, snr_db: float, rng: random.Random
) -> torch.Tensor:
    """
    Add white Gaussian noise to a waveform at a given SNR.

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        snr_db (float): Desired signal-to-noise ratio in dB.
        rng (random.Random): Seeded random number generator (unused here;
            torch.randn handles its own state after global seed is set).

    Returns:
        torch.Tensor: Noisy waveform, peak-renormalized to [-1, 1].
    """
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    out = waveform + noise
    peak = out.abs().max()
    return out / peak if peak > 0 else out


def random_gain(
    waveform: torch.Tensor, gain_range: list, rng: random.Random
) -> torch.Tensor:
    """
    Scale a waveform by a uniformly sampled gain factor.

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        gain_range (list): [min_gain, max_gain] for uniform sampling.
        rng (random.Random): Seeded random number generator.

    Returns:
        torch.Tensor: Gain-scaled waveform.
    """
    gain = rng.uniform(gain_range[0], gain_range[1])
    return waveform * gain


def time_shift(
    waveform: torch.Tensor, max_shift_ratio: float, rng: random.Random
) -> torch.Tensor:
    """
    Cyclically shift a waveform along the time axis.

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        max_shift_ratio (float): Maximum shift as a fraction of total length.
        rng (random.Random): Seeded random number generator.

    Returns:
        torch.Tensor: Cyclically shifted waveform.
    """
    n_samples = waveform.shape[-1]
    max_shift = int(n_samples * max_shift_ratio)
    shift = rng.randint(-max_shift, max_shift)
    return torch.roll(waveform, shift, dims=-1)


def augment_waveform(
    waveform: torch.Tensor, cfg: dict, rng: random.Random
) -> torch.Tensor:
    """
    Apply one randomly chosen augmentation technique to a waveform.

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        cfg (dict): Augmentation config with keys noise_snr_db, gain_range,
            max_shift_ratio.
        rng (random.Random): Seeded random number generator.

    Returns:
        torch.Tensor: Augmented waveform.
    """
    technique = rng.choice(["noise", "gain", "shift"])
    if technique == "noise":
        return add_noise(waveform, cfg["noise_snr_db"], rng)
    if technique == "gain":
        return random_gain(waveform, cfg["gain_range"], rng)
    return time_shift(waveform, cfg["max_shift_ratio"], rng)


def augment_dataset(config_path: str = "configs/base_config.yaml") -> None:
    """
    Generate one augmented copy per training clip and update train.csv.

    Skips clips whose augmented file already exists so the script is safely
    re-runnable. New rows are appended to train.csv with the same label and
    source_clip as the original, and filepath pointing to the ``_aug`` file.

    Args:
        config_path (str): Path to the base YAML config file.

    Returns:
        None
    """
    cfg = load_config(config_path)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    target_sr = cfg["audio"]["target_sr"]
    aug_cfg = cfg["augmentation"]

    # Fix all random seeds for reproducibility
    seed = aug_cfg["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    train_csv = splits_dir / "train.csv"
    train_df = pd.read_csv(train_csv)

    # Only augment original clips (skip rows already marked as augmented)
    originals = train_df[~train_df["filepath"].str.endswith("_aug.wav")]

    new_rows = []
    total = len(originals)
    print(f"Augmenting {total} training clips...")

    for i, row in originals.iterrows():
        src_path = Path(row["filepath"])
        aug_path = src_path.with_stem(src_path.stem + "_aug")

        if not aug_path.exists():
            waveform, _ = torchaudio.load(str(src_path))
            augmented = augment_waveform(waveform, aug_cfg, rng)
            torchaudio.save(str(aug_path), augmented, target_sr)

        new_rows.append(
            {
                "filepath": str(aug_path),
                "source_clip": row["source_clip"],
                "guitar_type": row["guitar_type"],
                "tempo": row["tempo"],
                "genre": row["genre"],
                "pedal": row["pedal"],
                "label": row["label"],
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} done")

    augmented_df = pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)
    augmented_df.to_csv(train_csv, index=False)

    print(f"Done. train.csv updated: {len(train_df)} -> {len(augmented_df)} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment training clips and update train.csv."
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    augment_dataset(args.config)
