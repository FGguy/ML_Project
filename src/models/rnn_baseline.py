"""
Single-layer RNN baseline for guitar pedal classification from raw audio.

The waveform is downsampled by striding before being fed into the RNN,
reducing the sequence length to a tractable number of steps. Variable-length
inputs are handled via packed sequences so padding never influences the final
hidden state.

Intentionally kept simple — this is a floor to beat, not a competitive model.

Reference: Damskägg et al., "Real-Time Modeling of Audio Distortion Circuits
with Deep Learning" (2019) informs the rationale for time-domain modelling.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNBaseline(nn.Module):
    """
    Single-layer RNN that classifies raw audio waveforms.

    The input waveform is downsampled by taking every ``downsample_factor``-th
    sample, then fed as a sequence into a vanilla RNN. The last valid hidden
    state is passed through a linear classifier.

    Args:
        downsample_factor (int): Step size for waveform downsampling.
        hidden_size (int): Number of hidden units in the RNN.
        num_classes (int): Number of output classes.
    """

    def __init__(self, downsample_factor: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.downsample_factor = downsample_factor
        # input_size=1 because each time step is a single amplitude value
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, 1, samples) — downsample along time axis
        x = x[:, :, :: self.downsample_factor]  # (batch, 1, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 1) for batch_first RNN

        if lengths is not None:
            ds_lengths = (
                (lengths + self.downsample_factor - 1) // self.downsample_factor
            ).clamp(min=1)
            packed = pack_padded_sequence(
                x, ds_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.rnn(packed)
        else:
            _, h_n = self.rnn(x)

        h_n = h_n.squeeze(0)  # (batch, hidden_size)
        return self.classifier(h_n)


def build_rnn(config: dict) -> RNNBaseline:
    """
    Instantiate an RNNBaseline from a config dictionary.

    Args:
        config (dict): Model config with keys downsample_factor, hidden_size,
            num_classes.

    Returns:
        RNNBaseline: Configured model instance.
    """
    return RNNBaseline(
        downsample_factor=config["downsample_factor"],
        hidden_size=config["hidden_size"],
        num_classes=config["num_classes"],
    )
