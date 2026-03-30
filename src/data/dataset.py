"""
PyTorch Dataset classes for the guitar pedal classification project.

Two classes are provided:
  - RawAudioDataset: loads pre-processed .wav files for the 1D CNN and RNN.
  - SpectrogramDataset: loads pre-computed .pt mel spectrograms for the CRDNN.

Both classes accept a split CSV path and return (tensor, label) pairs.
Features are loaded from disk on demand — nothing is recomputed at runtime.
"""

import warnings
from pathlib import Path

import pandas as pd
import torch
import torchaudio

from src.utils import load_config


class RawAudioDataset(torch.utils.data.Dataset):
    """
    Dataset that serves raw waveform tensors for the 1D CNN and RNN baseline.

    Loads .wav files on the fly and pads or crops them to a fixed number of
    samples so that all tensors in a batch share the same shape.

    Args:
        split_csv (str): Path to a split CSV (train.csv / val.csv / test.csv).
        config_path (str): Path to the base YAML config file.
    """

    def __init__(self, split_csv: str, config_path: str = "configs/base_config.yaml"):
        cfg = load_config(config_path)
        self.target_sr = cfg["audio"]["target_sr"]
        # Fixed length in samples: use fixed_time_frames * hop_length as a
        # consistent proxy so waveform and spectrogram lengths are aligned.
        mel_cfg = cfg["mel"]
        self.fixed_samples = mel_cfg["fixed_time_frames"] * mel_cfg["hop_length"]

        df = pd.read_csv(split_csv)
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, _ = torchaudio.load(self.filepaths[idx])

        waveform = self._pad_or_crop(waveform)
        return waveform, self.labels[idx]

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Crop or zero-pad a waveform to self.fixed_samples along the time axis.

        Args:
            waveform (torch.Tensor): Audio of shape (1, samples).

        Returns:
            torch.Tensor: Audio of shape (1, fixed_samples).
        """
        n = waveform.shape[-1]
        if n >= self.fixed_samples:
            return waveform[..., : self.fixed_samples]
        return torch.nn.functional.pad(waveform, (0, self.fixed_samples - n))


class SpectrogramDataset(torch.utils.data.Dataset):
    """
    Dataset that serves pre-computed mel spectrogram tensors for the CRDNN.

    Expects .pt files in data/processed/spectrograms/ produced by
    mel_precompute.py. Each tensor has shape (1, n_mels, fixed_time_frames)
    and is already normalized with train-set statistics.

    Args:
        split_csv (str): Path to a split CSV (train.csv / val.csv / test.csv).
        config_path (str): Path to the base YAML config file.
    """

    def __init__(self, split_csv: str, config_path: str = "configs/base_config.yaml"):
        cfg = load_config(config_path)
        self.spec_dir = Path(cfg["paths"]["processed_spectrograms_dir"])

        df = pd.read_csv(split_csv)
        self.stems = [Path(fp).stem for fp in df["filepath"].tolist()]
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        spec = torch.load(self.spec_dir / f"{self.stems[idx]}.pt", weights_only=True)
        return spec, self.labels[idx]
