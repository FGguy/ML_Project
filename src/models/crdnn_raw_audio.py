"""
1D audio CRDNN for guitar pedal classification from raw audio waveforms.

Strided Conv1d blocks compress the high-dimensional time-domain signal, then
a GRU captures temporal dependencies over the compressed sequence. The last
valid hidden state is used for classification.

Improvements applied:
- ELU activations replace ReLU (reduces dead-neuron risk on small datasets).
  Ref: Pons et al. (1703.06697)
- Residual (shortcut) connections in each conv block for training stability.
  Ref: Dai et al. (1610.00087)
- Progressive inter-block dropout (light early, heavier late).
  Ref: Rossi et al. (DAFx25_paper_16)
- Optional Global Average Pooling head instead of GRU (fully-convolutional).
  Ref: Dai et al. (1610.00087), Lee et al. SampleCNN (applsci-08-00150)

Reference: Damskägg et al., "Real-Time Modeling of Audio Distortion Circuits
with Deep Learning" (2019) — informs the rationale for time-domain modelling.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class ConvBlock1d(nn.Module):
    """
    1D conv block with residual shortcut: Conv1d → BatchNorm1d → ELU.

    A 1×1 projection shortcut is added when in_channels ≠ out_channels or
    stride > 1 so the residual can always be added element-wise.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size (should be odd).
        stride (int): Convolution stride (controls downsampling).
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
        )
        # Project the shortcut whenever shape changes so residual add is valid.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class AudioCRDNN(nn.Module):
    """
    Convolutional-Recurrent-DNN operating on raw audio waveforms.

    Conv1d blocks with striding compress the sequence length. A GRU (or
    optional global average pool) then produces a fixed-size representation
    for classification.

    Args:
        conv_blocks (list): List of [out_channels, kernel_size, stride] triples.
        gru_hidden (int): GRU hidden size (ignored when use_gap=True).
        dropout (float): Dropout probability before the classifier.
        inter_dropout_rates (list): Per-block inter-layer dropout rates; length
            must equal len(conv_blocks).
        num_classes (int): Number of output classes.
        use_gap (bool): If True, replace GRU with global average pooling.
    """

    def __init__(
        self,
        conv_blocks: list,
        gru_hidden: int,
        dropout: float,
        inter_dropout_rates: list,
        num_classes: int,
        use_gap: bool = False,
    ):
        super().__init__()

        self.use_gap = use_gap
        layers = []
        in_ch = 1
        for (out_ch, kernel, stride), drop_rate in zip(conv_blocks, inter_dropout_rates):
            layers.append(ConvBlock1d(in_ch, out_ch, kernel, stride))
            layers.append(nn.Dropout(drop_rate))
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        self._conv_strides = [s for _, _, s in conv_blocks]
        self._final_channels = in_ch

        if use_gap:
            # Fully convolutional head — no recurrent computation.
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_ch, num_classes),
            )
        else:
            self.gru = nn.GRU(in_ch, gru_hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(gru_hidden, num_classes),
            )

    def _cnn_output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute CNN output lengths from input lengths.

        With padding=k//2 and odd k, each conv gives ceil(L/stride).

        Args:
            lengths (torch.Tensor): Input lengths of shape (batch,).

        Returns:
            torch.Tensor: Output lengths after all conv blocks.
        """
        for stride in self._conv_strides:
            lengths = (lengths + stride - 1) // stride
        return lengths.clamp(min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, 1, L)
        x = self.encoder(x)  # (batch, channels, L')

        if self.use_gap:
            x = x.mean(dim=-1)  # global average pool → (batch, channels)
            return self.head(x)

        x = x.permute(0, 2, 1)  # (batch, L', channels) for GRU

        if lengths is not None:
            out_lengths = self._cnn_output_lengths(lengths)
            packed = pack_padded_sequence(
                x, out_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)

        h_n = h_n.squeeze(0)  # (batch, gru_hidden)
        return self.head(h_n)


def build_cnn1d(config: dict) -> AudioCRDNN:
    """
    Instantiate an AudioCRDNN from a config dictionary.

    Args:
        config (dict): Model config with keys conv_blocks, gru_hidden, dropout,
            inter_dropout_rates, num_classes, use_gap.

    Returns:
        AudioCRDNN: Configured model instance.
    """
    n_blocks = len(config["conv_blocks"])
    # Fall back to uniform zero rates if the key is absent (backward compat).
    inter_rates = config.get("inter_dropout_rates", [0.0] * n_blocks)
    return AudioCRDNN(
        conv_blocks=config["conv_blocks"],
        gru_hidden=config["gru_hidden"],
        dropout=config["dropout"],
        inter_dropout_rates=inter_rates,
        num_classes=config["num_classes"],
        use_gap=config.get("use_gap", False),
    )
