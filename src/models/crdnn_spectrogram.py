"""
CRDNN for guitar pedal classification from mel spectrograms.

Architecture: 2D CNN blocks extract local spectro-temporal features, a GRU
captures temporal dependencies across the compressed sequence, and a linear
head produces class logits.

Improvements applied:
- ELU activations replace ReLU.
  Ref: Pons et al. (1703.06697)
- Multi-scale parallel branches in the first conv block (1×9, 9×1, 3×3) to
  capture time-invariant spectral patterns, freq-invariant temporal patterns,
  and local joint patterns simultaneously.
  Ref: Pons et al. (1703.06697)
- Adaptive max-pool over the full frequency axis before the GRU forces the
  model to be pitch-agnostic.
  Ref: Pons et al. (1703.06697)
- Progressive spatial dropout (Dropout2d) between conv blocks.
  Ref: Rossi et al. (DAFx25_paper_16)

Reference: Piczak, "Environmental sound classification with convolutional
neural networks," MLSP 2015 — informs CNN-on-spectrogram design.
"""

import torch
import torch.nn as nn


class ConvBlock2d(nn.Module):
    """
    Single 2D conv block: Conv2d → BatchNorm2d → ELU → MaxPool2d.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Square convolution kernel size.
        pool_size (int): Pool factor applied to both freq and time axes.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(pool_size, pool_size)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleConvBlock2d(nn.Module):
    """
    Multi-scale first conv block with three parallel branches.

    Branch shapes (padding chosen so spatial dims are preserved before pooling):
      - 1×9  (time-axis filter): captures pitch-invariant temporal texture
      - 9×1  (freq-axis filter): captures duration-invariant spectral shape
      - 3×3  (square filter): captures local joint time-frequency patterns

    The three branches are concatenated along the channel dimension, giving
    out_channels total filters. out_channels must be divisible by 3.

    Args:
        in_channels (int): Number of input channels (typically 1).
        out_channels (int): Total output channels (must be divisible by 3).
        pool_size (int): Pool factor applied to both axes after concat.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_size: int):
        super().__init__()
        assert out_channels % 3 == 0, "out_channels must be divisible by 3 for multi-scale block"
        branch_ch = out_channels // 3

        self.branch_time = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=(1, 9), padding=(0, 4)),
            nn.BatchNorm2d(branch_ch),
            nn.ELU(),
        )
        self.branch_freq = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(branch_ch),
            nn.ELU(),
        )
        self.branch_square = nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(branch_ch),
            nn.ELU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=(pool_size, pool_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.branch_time(x)
        f = self.branch_freq(x)
        s = self.branch_square(x)
        return self.pool(torch.cat([t, f, s], dim=1))


class CRDNN(nn.Module):
    """
    Convolutional-Recurrent-DNN operating on mel spectrograms.

    Args:
        cnn_blocks (list): List of [out_channels, kernel_size, pool_size]
            triples, one per 2D conv block.
        gru_hidden (int): GRU hidden size.
        dropout (float): Dropout probability before the classifier.
        inter_dropout_rates (list): Spatial dropout rate after each conv block.
            Length must equal len(cnn_blocks).
        num_classes (int): Number of output classes.
        use_multi_scale_first (bool): If True, replace the first ConvBlock2d
            with a MultiScaleConvBlock2d.
    """

    def __init__(
        self,
        cnn_blocks: list,
        gru_hidden: int,
        dropout: float,
        inter_dropout_rates: list,
        num_classes: int,
        use_multi_scale_first: bool = True,
    ):
        super().__init__()

        layers = []
        in_ch = 1
        for i, ((out_ch, kernel, pool), drop_rate) in enumerate(
            zip(cnn_blocks, inter_dropout_rates)
        ):
            if i == 0 and use_multi_scale_first:
                layers.append(MultiScaleConvBlock2d(in_ch, out_ch, pool))
            else:
                layers.append(ConvBlock2d(in_ch, out_ch, kernel, pool))
            # Spatial dropout zeros entire feature maps — more effective than
            # per-element dropout for convolutional activations.
            layers.append(nn.Dropout2d(drop_rate))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.gru_hidden = gru_hidden
        self.gru = None  # built lazily on first forward pass
        self.dropout = nn.Dropout(dropout)
        self.classifier = None
        self._num_classes = num_classes

    def _build_rnn_head(self, n_channels: int) -> None:
        """
        Build the GRU and classifier once CNN output size is known.

        With the freq-axis max-pool the GRU input equals the channel count,
        not channels × freq_bins.

        Args:
            n_channels (int): Number of channels after freq-axis collapse.
        """
        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=self.gru_hidden,
            batch_first=True,
        )
        self.classifier = nn.Linear(self.gru_hidden, self._num_classes)
        device = next(self.cnn.parameters()).device
        self.gru = self.gru.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time)
        x = self.cnn(x)  # (batch, channels, freq, time)

        # Collapse freq axis via adaptive max-pool for pitch invariance.
        # Ref: Pons et al. (1703.06697)
        x = x.max(dim=2).values  # (batch, channels, time)
        x = x.permute(0, 2, 1)   # (batch, time, channels)

        if self.gru is None:
            self._build_rnn_head(x.shape[-1])

        _, h_n = self.gru(x)  # h_n: (1, batch, gru_hidden)
        h_n = h_n.squeeze(0)  # (batch, gru_hidden)
        return self.classifier(self.dropout(h_n))


def build_crdnn(config: dict) -> CRDNN:
    """
    Instantiate a CRDNN from a config dictionary.

    Args:
        config (dict): Model config with keys cnn_blocks, gru_hidden, dropout,
            inter_dropout_rates, num_classes, use_multi_scale_first.

    Returns:
        CRDNN: Configured model instance.
    """
    n_blocks = len(config["cnn_blocks"])
    inter_rates = config.get("inter_dropout_rates", [0.0] * n_blocks)
    return CRDNN(
        cnn_blocks=config["cnn_blocks"],
        gru_hidden=config["gru_hidden"],
        dropout=config["dropout"],
        inter_dropout_rates=inter_rates,
        num_classes=config["num_classes"],
        use_multi_scale_first=config.get("use_multi_scale_first", True),
    )
