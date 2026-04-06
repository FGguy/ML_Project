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
from torch.nn.utils.rnn import pad_sequence

from src.data.transforms import SpecAugment
from src.utils import load_config


class RawAudioDataset(torch.utils.data.Dataset):
    """
    Dataset that serves raw waveform tensors for the 1D CNN and RNN baseline.

    Clips are returned at their original length — variable-length batching is
    handled by the ``collate_variable_length`` collate function.

    Args:
        split_csv (str): Path to a split CSV (train.csv / val.csv / test.csv).
        config_path (str): Path to the base YAML config file.
    """

    def __init__(self, split_csv: str, config_path: str = "configs/base_config.yaml"):
        cfg = load_config(config_path)
        self.target_sr = cfg["audio"]["target_sr"]

        df = pd.read_csv(split_csv)
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, _ = torchaudio.load(self.filepaths[idx])
        return waveform, self.labels[idx]


def collate_variable_length(
    batch: list,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate waveforms of variable length into a padded batch.

    Waveforms are zero-padded to the longest sequence in the batch. The
    original lengths are returned so models can use packed sequences and
    avoid attending to padding.

    Args:
        batch (list): List of (waveform, label) tuples where each waveform
            has shape (1, L) with potentially different L.

    Returns:
        tuple: (padded_waveforms, labels, lengths)
            - padded_waveforms: (batch, 1, max_L)
            - labels: (batch,)
            - lengths: (batch,) original sample counts, dtype=torch.long
    """
    waveforms, labels = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)
    # pad_sequence expects (L,) tensors; squeeze channel, pad, add it back
    padded = pad_sequence(
        [w.squeeze(0) for w in waveforms], batch_first=True
    ).unsqueeze(1)  # (batch, 1, max_L)
    return padded, torch.tensor(labels), lengths


class SpectrogramDataset(torch.utils.data.Dataset):
    """
    Dataset that serves pre-computed mel spectrogram tensors for the CRDNN.

    Expects .pt files in data/processed/spectrograms/ produced by
    mel_precompute.py. Each tensor has shape (1, n_mels, fixed_time_frames)
    and is already normalized with train-set statistics.

    When ``train=True`` and spec_augment_cfg is provided, SpecAugment is
    applied on-the-fly — frequency and time masking to improve generalisation.
    Val and test splits must use ``train=False`` so augmentation is never
    applied at evaluation time.

    Args:
        split_csv (str): Path to a split CSV (train.csv / val.csv / test.csv).
        config_path (str): Path to the base YAML config file.
        train (bool): If True, apply SpecAugment (training split only).
        spec_augment_cfg (dict | None): SpecAugment hyper-parameters. Expected
            keys: freq_mask_param, time_mask_param, num_freq_masks,
            num_time_masks. If None, augmentation is skipped even when
            train=True.
    """

    def __init__(
        self,
        split_csv: str,
        config_path: str = "configs/base_config.yaml",
        train: bool = False,
        spec_augment_cfg: dict = None,
    ):
        cfg = load_config(config_path)
        self.spec_dir = Path(cfg["paths"]["processed_spectrograms_dir"])

        df = pd.read_csv(split_csv)
        self.stems = [Path(fp).stem for fp in df["filepath"].tolist()]
        self.labels = df["label"].tolist()

        self.augment = None
        if train and spec_augment_cfg is not None:
            self.augment = SpecAugment(
                freq_mask_param=spec_augment_cfg["freq_mask_param"],
                time_mask_param=spec_augment_cfg["time_mask_param"],
                num_freq_masks=spec_augment_cfg.get("num_freq_masks", 1),
                num_time_masks=spec_augment_cfg.get("num_time_masks", 2),
            )

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        spec = torch.load(self.spec_dir / f"{self.stems[idx]}.pt", weights_only=True)
        if self.augment is not None:
            spec = self.augment(spec)
        return spec, self.labels[idx]
