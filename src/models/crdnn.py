"""
CRDNN for guitar pedal classification from mel spectrograms.

Architecture: 2D CNN blocks extract local spectro-temporal features, a GRU
captures temporal dependencies across the compressed sequence, and a linear
head produces class logits.

The CNN progressively halves the frequency dimension via MaxPool2d while
preserving the time axis. After the CNN the frequency bins are collapsed into
the channel dimension so the GRU sees one vector per time frame.

Reference: Piczak, "Environmental sound classification with convolutional
neural networks," MLSP 2015 — informs CNN-on-spectrogram design.
"""

import torch
import torch.nn as nn


class ConvBlock2d(nn.Module):
    """
    Single 2D conv block: Conv2d → BatchNorm2d → ReLU → MaxPool2d.

    Pooling is applied only along the frequency axis (pool kernel 2×1) to
    preserve the full time resolution for the GRU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Square convolution kernel size.
        pool_size (int): Frequency-axis pool factor.
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
            nn.ReLU(),
            # Pool only the frequency axis to keep full time resolution for GRU
            nn.MaxPool2d(kernel_size=(pool_size, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CRDNN(nn.Module):
    """
    Convolutional-Recurrent-DNN operating on mel spectrograms.

    Args:
        cnn_blocks (list): List of [out_channels, kernel_size, pool_size]
            triples, one per 2D conv block.
        gru_hidden (int): GRU hidden size.
        dropout (float): Dropout probability before the classifier.
        num_classes (int): Number of output classes.
    """

    def __init__(
        self, cnn_blocks: list, gru_hidden: int, dropout: float, num_classes: int
    ):
        super().__init__()

        layers = []
        in_ch = 1
        for out_ch, kernel, pool in cnn_blocks:
            layers.append(ConvBlock2d(in_ch, out_ch, kernel, pool))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # GRU input size is determined at forward time after collapsing freq dim
        self.gru_hidden = gru_hidden
        self.gru = None  # built lazily on first forward pass
        self._gru_input_size = None

        self.dropout = nn.Dropout(dropout)
        self.classifier = None  # built lazily
        self._num_classes = num_classes

    def _build_rnn_head(self, freq_channels: int) -> None:
        """
        Build the GRU and classifier once the CNN output size is known.

        Args:
            freq_channels (int): Number of features per time step after
                flattening the frequency and channel dimensions.
        """
        self._gru_input_size = freq_channels
        self.gru = nn.GRU(
            input_size=freq_channels,
            hidden_size=self.gru_hidden,
            batch_first=True,
        )
        self.classifier = nn.Linear(self.gru_hidden, self._num_classes)
        # Move to same device as cnn parameters
        device = next(self.cnn.parameters()).device
        self.gru = self.gru.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time)
        x = self.cnn(x)  # (batch, channels, freq, time)

        batch, channels, freq, time = x.shape
        # Collapse channels and freq into a single feature dim per time step
        x = x.permute(0, 3, 1, 2).reshape(batch, time, channels * freq)

        if self.gru is None:
            self._build_rnn_head(channels * freq)

        _, h_n = self.gru(x)  # h_n: (1, batch, gru_hidden)
        h_n = h_n.squeeze(0)  # (batch, gru_hidden)
        return self.classifier(self.dropout(h_n))


def build_crdnn(config: dict) -> CRDNN:
    """
    Instantiate a CRDNN from a config dictionary.

    Args:
        config (dict): Model config with keys cnn_blocks, gru_hidden,
            dropout, num_classes.

    Returns:
        CRDNN: Configured model instance.
    """
    return CRDNN(
        cnn_blocks=config["cnn_blocks"],
        gru_hidden=config["gru_hidden"],
        dropout=config["dropout"],
        num_classes=config["num_classes"],
    )
