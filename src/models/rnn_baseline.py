"""
Single-layer RNN baseline for guitar pedal classification from raw audio.

The waveform is downsampled by striding before being fed into the RNN,
reducing the sequence length to match the mel spectrogram's temporal
resolution (~fixed_time_frames steps). This makes the baseline computationally
tractable while operating in the time domain.

Intentionally kept simple — this is a floor to beat, not a competitive model.

Reference: Damskägg et al., "Real-Time Modeling of Audio Distortion Circuits
with Deep Learning" (2019) informs the rationale for time-domain modelling.
"""

import torch
import torch.nn as nn


class RNNBaseline(nn.Module):
    """
    Single-layer RNN that classifies raw audio waveforms.

    The input waveform is downsampled by taking every ``downsample_factor``-th
    sample, then fed as a sequence into a vanilla RNN. The last hidden state
    is passed through a linear classifier.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, samples) — downsample along time axis
        x = x[:, :, :: self.downsample_factor]  # (batch, 1, seq_len)
        x = x.permute(0, 2, 1)  # (batch, seq_len, 1) for batch_first RNN
        _, h_n = self.rnn(x)  # h_n: (1, batch, hidden_size)
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
