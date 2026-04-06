"""
On-the-fly spectrogram augmentation transforms.

SpecAugment masks randomly selected frequency bins and time frames in the
mel spectrogram during training. It is more suitable for audio than image-
style operations (flip, rotation) which destroy the time-frequency semantics.

Applied to SpectrogramDataset training samples only — val and test are left
untouched.

Reference: Ferreira-Paiva et al. (paper_5085), Purwins et al. (1905.00078)
"""

import random

import torch


class SpecAugment:
    """
    Randomly mask frequency and time bands in a mel spectrogram tensor.

    Implements the frequency and time masking steps from SpecAugment. Each
    call applies ``num_freq_masks`` independent frequency masks and
    ``num_time_masks`` independent time masks. Masked values are set to the
    spectrogram mean so the model cannot infer content from unusual values.

    Args:
        freq_mask_param (int): Maximum number of consecutive mel bins to mask.
        time_mask_param (int): Maximum number of consecutive time frames to mask.
        num_freq_masks (int): Number of independent frequency mask applications.
        num_time_masks (int): Number of independent time mask applications.
    """

    def __init__(
        self,
        freq_mask_param: int,
        time_mask_param: int,
        num_freq_masks: int = 1,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment masking to a single spectrogram tensor.

        Args:
            spec (torch.Tensor): Shape (1, n_mels, time_frames).

        Returns:
            torch.Tensor: Masked spectrogram, same shape as input.
        """
        spec = spec.clone()
        mean_val = spec.mean().item()
        _, n_mels, n_time = spec.shape

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if f == 0 or n_mels <= f:
                continue
            f0 = random.randint(0, n_mels - f)
            spec[:, f0: f0 + f, :] = mean_val

        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if t == 0 or n_time <= t:
                continue
            t0 = random.randint(0, n_time - t)
            spec[:, :, t0: t0 + t] = mean_val

        return spec
