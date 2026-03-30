"""
Assign each source clip to exactly one pedal class and apply the effect.

The 507 source clips are divided into 6 balanced groups using stratified
round-robin assignment over (guitar_type, genre, tempo) strata with a fixed
seed. Each clip is then processed with its assigned pedal chain via the
pedalboard library and written to data/processed/audio/.

Run this after preprocess.py and before split.py.
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import soundfile as sf
from pedalboard import Clipping, Distortion, Gain, Pedalboard

from src.utils import load_config


def build_pedalboard(pedal_cfg: dict) -> Pedalboard:
    """
    Construct a pedalboard.Pedalboard from a pedal config dict.

    Supported types:
        - ``distortion``: single Distortion plugin with drive_db.
        - ``distortion_clipping``: Distortion followed by Clipping.
        - ``gain``: single Gain plugin with gain_db.

    Args:
        pedal_cfg (dict): Pedal configuration with at minimum a ``type`` key.

    Returns:
        Pedalboard: Configured pedalboard ready to process audio.

    Raises:
        ValueError: If ``type`` is not one of the supported values.
    """
    pedal_type = pedal_cfg["type"]
    if pedal_type == "distortion":
        return Pedalboard([Distortion(drive_db=pedal_cfg["drive_db"])])
    if pedal_type == "distortion_clipping":
        return Pedalboard(
            [
                Distortion(drive_db=pedal_cfg["drive_db"]),
                Clipping(threshold_db=pedal_cfg["threshold_db"]),
            ]
        )
    if pedal_type == "gain":
        return Pedalboard([Gain(gain_db=pedal_cfg["gain_db"])])
    raise ValueError(f"Unknown pedal type: {pedal_type!r}")


def assign_pedals(df: pd.DataFrame, pedal_names: list[str], seed: int) -> pd.Series:
    """
    Assign each clip in df to one pedal class using stratified round-robin.

    Clips are grouped by (guitar_type, genre, tempo). Within each stratum the
    clips are shuffled, then appended to a global ordering. The pedal label
    cycles over the full ordering with a counter that carries across strata,
    so the final class sizes stay as equal as possible regardless of stratum
    sizes.

    Args:
        df (pd.DataFrame): Flat index with columns guitar_type, genre, tempo.
        pedal_names (list[str]): Ordered list of pedal class names.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.Series: Pedal name per row, indexed to match df.
    """
    rng = random.Random(seed)
    assignments = pd.Series(index=df.index, dtype=str)

    # Collect all indices stratum by stratum (shuffled within each)
    ordered_indices = []
    strata = df.groupby(["guitar_type", "genre", "tempo"])
    for _, group in strata:
        indices = group.index.tolist()
        rng.shuffle(indices)
        ordered_indices.extend(indices)

    # Single round-robin pass over all clips
    for counter, idx in enumerate(ordered_indices):
        assignments[idx] = pedal_names[counter % len(pedal_names)]

    return assignments


def apply_distortion(
    src_path: Path, dst_path: Path, board: Pedalboard, sample_rate: int
) -> None:
    """
    Apply a pedalboard effect chain to a .wav file and write the result.

    Args:
        src_path (Path): Path to the source .wav file.
        dst_path (Path): Destination path for the processed .wav file.
        board (Pedalboard): Configured pedalboard to apply.
        sample_rate (int): Sample rate of the audio (used by pedalboard).

    Returns:
        None
    """
    audio, sr = sf.read(str(src_path), dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); pedalboard expects (channels, samples)
    audio = audio.T
    processed = board(audio, sr)
    sf.write(str(dst_path), processed.T, sr)


def run_distortion(
    base_config: str = "configs/base_config.yaml",
    distortion_config: str = "configs/distortion_config.yaml",
) -> None:
    """
    Main entry point: assign pedals, apply effects, write audio_index.csv.

    Args:
        base_config (str): Path to the base YAML config file.
        distortion_config (str): Path to the distortion YAML config file.

    Returns:
        None
    """
    base_cfg = load_config(base_config)
    dist_cfg = load_config(distortion_config)

    flat_dir = Path(base_cfg["paths"]["raw_flat_dir"])
    out_dir = Path(base_cfg["paths"]["processed_audio_dir"])
    index_out = Path(base_cfg["paths"]["audio_index"])
    target_sr = base_cfg["audio"]["target_sr"]

    out_dir.mkdir(parents=True, exist_ok=True)

    pedal_cfgs = dist_cfg["pedals"]
    label_map = dist_cfg["labels"]
    seed = dist_cfg["assignment_seed"]
    pedal_names = list(pedal_cfgs.keys())

    df = pd.read_csv(base_cfg["paths"]["flat_index"])

    print("Assigning pedal classes...")
    df["pedal"] = assign_pedals(df, pedal_names, seed)
    df["label"] = df["pedal"].map(label_map)

    # Log class balance
    counts = df["pedal"].value_counts().sort_index()
    print("Class distribution:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    records = []
    total = len(df)
    print(f"\nApplying distortion to {total} clips...")

    for i, row in df.iterrows():
        pedal_name = row["pedal"]
        board = build_pedalboard(pedal_cfgs[pedal_name])

        src = flat_dir / row["flat_filename"]
        stem = Path(row["flat_filename"]).stem
        dst_name = f"{stem}__{pedal_name}.wav"
        dst = out_dir / dst_name

        if not dst.exists():
            apply_distortion(src, dst, board, target_sr)

        records.append(
            {
                "filepath": str(dst),
                "source_clip": row["flat_filename"],
                "guitar_type": row["guitar_type"],
                "tempo": row["tempo"],
                "genre": row["genre"],
                "pedal": pedal_name,
                "label": row["label"],
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{total} done")

    audio_df = pd.DataFrame(records)
    audio_df.to_csv(index_out, index=False)
    print(f"\nWrote audio index to {index_out} ({len(audio_df)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign pedal classes and apply distortion effects."
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--distortion-config", default="configs/distortion_config.yaml")
    args = parser.parse_args()
    run_distortion(args.config, args.distortion_config)
