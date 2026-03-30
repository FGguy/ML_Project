"""
Flatten the raw IDMT-SMT dataset into a single directory with a metadata CSV.

Walks data/raw/ and copies every .wav file to data/raw_flat/, renaming each
file so that guitar_type, tempo, and genre are encoded in the filename. A
companion CSV (flat_index.csv) records all metadata extracted from the
original path and filename.
"""

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd

from src.utils import load_config


def parse_filename(filename: str) -> tuple[str, int]:
    """
    Parse the BPM and sample number from an IDMT-SMT audio filename.

    Expected format: ``{genre}_{sample_number}_{bpm}BPM.wav``

    Args:
        filename (str): Base filename without directory (e.g. "classical_1_80BPM.wav").

    Returns:
        tuple[str, int]: (sample_number as str, bpm as int).
            Returns ("unknown", 0) if the pattern does not match.
    """
    # Pattern: anything_NUMBER_NUMBERBPM.wav
    match = re.search(r"_(\d+)_(\d+)BPM\.wav$", filename, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    return "unknown", 0


def make_flat_filename(
    guitar_type: str, tempo: str, genre: str, original_name: str
) -> str:
    """
    Build the flat filename by encoding path-level metadata into the name.

    Uses double underscores as separators so that guitar_type values that
    contain spaces (e.g. "Career SG") survive the encoding without ambiguity.

    Args:
        guitar_type (str): Guitar type folder name (e.g. "acoustic_mic").
        tempo (str): Tempo folder name ("fast" or "slow").
        genre (str): Genre folder name (e.g. "classical").
        original_name (str): Original filename including extension.

    Returns:
        str: Flat filename, e.g. "acoustic_mic__fast__classical__classical_1_80BPM.wav".
    """
    return f"{guitar_type}__{tempo}__{genre}__{original_name}"


def collect_wav_files(raw_dir: Path) -> list[dict]:
    """
    Recursively find all .wav files under raw_dir and parse their metadata.

    Expected directory structure:
        raw_dir / guitar_type / tempo / genre / audio / filename.wav

    Args:
        raw_dir (Path): Root of the raw dataset directory.

    Returns:
        list[dict]: One dict per file with keys:
            flat_filename, original_path, guitar_type, tempo, genre,
            sample_number, bpm.
    """
    records = []
    for wav_path in sorted(raw_dir.rglob("*.wav")):
        # Expected: raw_dir / guitar_type / tempo / genre / audio / name.wav
        parts = wav_path.relative_to(raw_dir).parts
        if len(parts) < 5 or parts[3] != "audio":
            # Skip any file that doesn't fit the expected structure
            continue

        guitar_type = parts[0]
        tempo = parts[1]
        genre = parts[2]
        original_name = parts[4]

        sample_number, bpm = parse_filename(original_name)
        flat_filename = make_flat_filename(guitar_type, tempo, genre, original_name)

        records.append(
            {
                "flat_filename": flat_filename,
                "original_path": str(wav_path),
                "guitar_type": guitar_type,
                "tempo": tempo,
                "genre": genre,
                "sample_number": sample_number,
                "bpm": bpm,
            }
        )
    return records


def flatten_dataset(config_path: str = "configs/base_config.yaml") -> None:
    """
    Main entry point: copy all raw .wav files to a flat directory and write the index CSV.
    # noqa: E501

    Args:
        config_path (str): Path to the base YAML config file.

    Returns:
        None
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    flat_dir = Path(cfg["paths"]["raw_flat_dir"])
    index_path = Path(cfg["paths"]["flat_index"])

    flat_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {raw_dir} for .wav files...")
    records = collect_wav_files(raw_dir)
    print(f"Found {len(records)} .wav files.")

    for rec in records:
        src = Path(rec["original_path"])
        dst = flat_dir / rec["flat_filename"]
        if not dst.exists():
            shutil.copy2(src, dst)

    df = pd.DataFrame(records)
    df.to_csv(index_path, index=False)
    print(f"Wrote flat index to {index_path} ({len(df)} rows).")
    print(f"Files copied to {flat_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten IDMT-SMT dataset into a single directory."
    )
    parser.add_argument(
        "--config",
        default="configs/base_config.yaml",
        help="Path to the base YAML config file.",
    )
    args = parser.parse_args()
    flatten_dataset(args.config)
