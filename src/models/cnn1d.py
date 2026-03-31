"""
1D CNN for guitar pedal classification from raw audio waveforms.

Uses strided convolutions to progressively reduce the high-dimensional
time-domain input (580,864 samples) to a compact representation before
global average pooling and classification. Striding is the explicit
strategy called for in the proposal to handle raw audio dimensionality.

Reference: Damskägg et al., "Real-Time Modeling of Audio Distortion Circuits
with Deep Learning" (2019) — informs the rationale for time-domain modelling.
"""

import torch
import torch.nn as nn


class ConvBlock1d(nn.Module):
    """
    Single 1D conv block: Conv1d → BatchNorm1d → ReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        stride (int): Convolution stride (controls downsampling).
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1d(nn.Module):
    """
    Stack of strided 1D conv blocks followed by global average pooling and a
    linear classifier.

    Args:
        conv_blocks (list): List of [out_channels, kernel_size, stride] triples,
            one per conv block.
        dropout (float): Dropout probability before the classifier.
        num_classes (int): Number of output classes.
    """

    def __init__(self, conv_blocks: list, dropout: float, num_classes: int):
        super().__init__()

        layers = []
        in_ch = 1
        for out_ch, kernel, stride in conv_blocks:
            layers.append(ConvBlock1d(in_ch, out_ch, kernel, stride))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, samples)
        x = self.encoder(x)  # (batch, channels, reduced_len)
        x = x.mean(dim=-1)  # global average pool → (batch, channels)
        return self.classifier(x)


def build_cnn1d(config: dict) -> CNN1d:
    """
    Instantiate a CNN1d from a config dictionary.

    Args:
        config (dict): Model config with keys conv_blocks, dropout, num_classes.

    Returns:
        CNN1d: Configured model instance.
    """
    return CNN1d(
        conv_blocks=config["conv_blocks"],
        dropout=config["dropout"],
        num_classes=config["num_classes"],
    )
