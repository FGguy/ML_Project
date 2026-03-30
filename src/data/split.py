"""
Train / val / test split for the processed audio dataset.

Splits are performed at the source_clip level so that all processed variants
of the same clip land in the same split. The split is stratified by pedal
label to preserve class proportions. Split CSVs are written immediately to
data/splits/ and are the single source of truth for all downstream steps.

Run this after distortion.py and before normalize.py.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_config


def split_dataset(config_path: str = "configs/base_config.yaml") -> None:
    """
    Create grouped, stratified train/val/test splits and save them as CSVs.

    Each unique source_clip is assigned to exactly one split. Stratification
    is by pedal label so class proportions are preserved. The random seed and
    final class counts are printed for reproducibility verification.

    Args:
        config_path (str): Path to the base YAML config file.

    Returns:
        None
    """
    cfg = load_config(config_path)
    audio_index = Path(cfg["paths"]["audio_index"])
    splits_dir = Path(cfg["paths"]["splits_dir"])
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]
    seed = cfg["split"]["seed"]

    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(audio_index)

    # One row per source clip — the label to stratify on
    clip_labels = df[["source_clip", "label"]].drop_duplicates("source_clip")

    # First cut: train vs (val + test)
    val_test_ratio = val_ratio + (1.0 - train_ratio - val_ratio)
    train_clips, val_test_clips = train_test_split(
        clip_labels,
        test_size=val_test_ratio,
        stratify=clip_labels["label"],
        random_state=seed,
    )

    # Second cut: val vs test (equal halves of the remaining pool)
    val_clips, test_clips = train_test_split(
        val_test_clips,
        test_size=0.5,
        stratify=val_test_clips["label"],
        random_state=seed,
    )

    # Verify no source_clip overlap between splits
    assert set(train_clips["source_clip"]).isdisjoint(
        val_clips["source_clip"]
    ), "Leakage: train and val share source clips"
    assert set(train_clips["source_clip"]).isdisjoint(
        test_clips["source_clip"]
    ), "Leakage: train and test share source clips"
    assert set(val_clips["source_clip"]).isdisjoint(
        test_clips["source_clip"]
    ), "Leakage: val and test share source clips"

    train_df = df[df["source_clip"].isin(train_clips["source_clip"])]
    val_df = df[df["source_clip"].isin(val_clips["source_clip"])]
    test_df = df[df["source_clip"].isin(test_clips["source_clip"])]

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"Seed: {seed}")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    print("\nClass counts per split:")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        counts = split_df["pedal"].value_counts().sort_index()
        print(f"  {split_name}: {dict(counts)}")

    print(f"\nSaved splits to {splits_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create grouped stratified train/val/test splits."
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    args = parser.parse_args()
    split_dataset(args.config)
