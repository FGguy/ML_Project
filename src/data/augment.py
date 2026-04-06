"""
Data augmentation for the training split only.

For each clip in train.csv two augmentation techniques are randomly chosen
from the pool {noise, gain, shift, stretch} and applied in sequence.
Applying combinations rather than a single technique yields higher accuracy
than any single method alone.

Ref: Tsalera et al. (jsan-14-00091) — combinations outperform single methods;
     optimal dataset expansion is 50–100% (current script keeps 1:1 ratio).

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
import torch.nn.functional as F

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


def pitch_shift(
    waveform: torch.Tensor, n_steps: int, sample_rate: int
) -> torch.Tensor:
    """
    Shift the pitch of a waveform by a fixed number of semitones.

    Pedal distortion character (clipping threshold, harmonic structure) is
    largely pitch-invariant, so a pitch-shifted clip belongs to the same
    class and adds genuine timbral diversity to training.

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        n_steps (int): Number of semitones to shift (positive = higher pitch).
        sample_rate (int): Sample rate of the audio.

    Returns:
        torch.Tensor: Pitch-shifted waveform, same shape as input.
    """
    # Ref: Ferreira-Paiva et al. (paper_5085), Tsalera et al. (jsan-14-00091)
    return torchaudio.functional.pitch_shift(waveform, sample_rate, n_steps)


def time_stretch(
    waveform: torch.Tensor, rate: float, sample_rate: int
) -> torch.Tensor:
    """
    Stretch or compress a waveform's duration via speed perturbation.

    Resamples from ``int(sample_rate * rate)`` to ``sample_rate``, which
    changes both tempo and pitch by the same factor. For small rates (0.85–1.15)
    the pitch change is minor and the added timing diversity outweighs it. The
    output is trimmed or zero-padded to preserve the original length.

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        rate (float): Speed factor. >1 compresses (shorter), <1 stretches
            (longer).
        sample_rate (int): Sample rate of the audio.

    Returns:
        torch.Tensor: Speed-perturbed waveform, same shape as input.
    """
    # Ref: Ferreira-Paiva et al. (paper_5085), Tsalera et al. (jsan-14-00091)
    orig_len = waveform.shape[-1]
    new_sr = max(1, int(sample_rate * rate))
    stretched = torchaudio.functional.resample(waveform, new_sr, sample_rate)
    if stretched.shape[-1] > orig_len:
        return stretched[..., :orig_len]
    if stretched.shape[-1] < orig_len:
        pad_len = orig_len - stretched.shape[-1]
        return F.pad(stretched, (0, pad_len))
    return stretched


def augment_waveform(
    waveform: torch.Tensor, cfg: dict, rng: random.Random, sample_rate: int
) -> torch.Tensor:
    """
    Apply a random combination of two augmentation techniques to a waveform.

    Two techniques are sampled without replacement from the pool
    {noise, gain, shift, stretch}. Combining techniques yields better
    generalisation than any single technique in isolation.

    Ref: Tsalera et al. (jsan-14-00091)

    Args:
        waveform (torch.Tensor): Input audio of shape (channels, samples).
        cfg (dict): Augmentation config with keys noise_snr_db, gain_range,
            max_shift_ratio, pitch_shift_steps, time_stretch_rates.
        rng (random.Random): Seeded random number generator.
        sample_rate (int): Sample rate of the audio.

    Returns:
        torch.Tensor: Augmented waveform.
    """
    # pitch_shift is excluded here — it uses a full STFT phase vocoder and is
    # orders of magnitude slower than the other ops. The CRDNN gains pitch
    # invariance from the freq-axis max-pool; raw audio models benefit from
    # natural pitch variety in the dataset.
    pool = ["noise", "gain", "shift", "stretch"]
    techniques = rng.sample(pool, k=2)

    out = waveform
    for technique in techniques:
        if technique == "noise":
            out = add_noise(out, cfg["noise_snr_db"], rng)
        elif technique == "gain":
            out = random_gain(out, cfg["gain_range"], rng)
        elif technique == "shift":
            out = time_shift(out, cfg["max_shift_ratio"], rng)
        elif technique == "pitch":
            n_steps = rng.choice(cfg["pitch_shift_steps"])
            out = pitch_shift(out, n_steps, sample_rate)
        elif technique == "stretch":
            rate = rng.choice(cfg["time_stretch_rates"])
            out = time_stretch(out, rate, sample_rate)
    return out


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

    seed = aug_cfg["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)

    train_csv = splits_dir / "train.csv"
    train_df = pd.read_csv(train_csv)

    originals = train_df[~train_df["filepath"].str.endswith("_aug.wav")]

    new_rows = []
    total = len(originals)
    print(f"Augmenting {total} training clips...")

    for i, row in originals.iterrows():
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
