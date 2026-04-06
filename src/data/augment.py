"""
Data augmentation for the training split only.

For each clip selected for augmentation, two techniques are randomly chosen
from the pool {noise, gain, shift} and applied in sequence.
Applying combinations rather than a single technique yields higher accuracy
than any single method alone.

Ref: Tsalera et al. (jsan-14-00091) — combinations outperform single methods;
     optimal dataset expansion is 50–100% (controlled via aug_ratio in config).

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
        rng (random.Random): Seeded RNG (torch.randn uses global state).

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
    waveform: torch.Tensor, cfg: dict, rng: random.Random, sample_rate: int
) -> torch.Tensor:
    """
    Apply a random combination of two augmentation techniques to a waveform.

    Two techniques are sampled without replacement from the pool
    {noise, gain, shift}. Combining techniques yields better generalisation
    than any single technique in isolation.

    Ref: Tsalera et al. (jsan-14-00091)

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        cfg (dict): Augmentation config with keys noise_snr_db, gain_range,
            max_shift_ratio.
        rng (random.Random): Seeded random number generator.
        sample_rate (int): Sample rate of the audio.

    Returns:
        torch.Tensor: Augmented waveform.
    """
    # stretch (speed perturbation via resample) is excluded — it runs a full
    # polyphase filter per clip and is the main augmentation bottleneck.
    pool = ["noise", "gain", "shift"]
    techniques = rng.sample(pool, k=2)

    out = waveform
    for technique in techniques:
        if technique == "noise":
            out = add_noise(out, cfg["noise_snr_db"], rng)
        elif technique == "gain":
            out = random_gain(out, cfg["gain_range"], rng)
        elif technique == "shift":
            out = time_shift(out, cfg["max_shift_ratio"], rng)
    return out


def augment_dataset(config_path: str = "configs/base_config.yaml") -> None:
    """
    Generate augmented copies of a fraction of training clips and update train.csv.

    The fraction is controlled by ``aug_ratio`` in the augmentation config
    (e.g. 0.5 augments half the training clips, adding 50% more samples).
    Clips are sampled without replacement using the configured seed.

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

    seed = aug_cfg["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    train_csv = splits_dir / "train.csv"
    train_df = pd.read_csv(train_csv)

    originals = train_df[~train_df["filepath"].str.endswith("_aug.wav")]

    # Sample a stratified subset to augment based on aug_ratio
    aug_ratio = aug_cfg.get("aug_ratio", 1.0)
    n_to_aug = max(1, int(len(originals) * aug_ratio))
    to_augment = originals.sample(n=n_to_aug, random_state=seed)

    new_rows = []
    total = len(to_augment)
    print(f"Augmenting {total}/{len(originals)} training clips (aug_ratio={aug_ratio})...")

    for idx, (_, row) in enumerate(to_augment.iterrows()):
        src_path = Path(row["filepath"])
        aug_path = src_path.with_stem(src_path.stem + "_aug")

        if not aug_path.exists():
            waveform, _ = torchaudio.load(str(src_path))
            augmented = augment_waveform(waveform, aug_cfg, rng, target_sr)
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

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{total} done")

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
