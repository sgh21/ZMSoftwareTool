"""三通道时域 TCN，沿用原研究仓库的网络结构，不附带预训练权重。"""

from torch import nn
from torch.nn import functional as F
import torch


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, dilation: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden_channels * 2, 1)
        self.depthwise = nn.Conv1d(
            hidden_channels, hidden_channels, 3, padding=dilation,
            dilation=dilation, groups=hidden_channels,
        )
        self.output = nn.Conv1d(hidden_channels, channels, 1)
        self.norm = nn.GroupNorm(1, hidden_channels)

    def forward(self, signal):
        value, gate = self.input(signal).chunk(2, dim=1)
        hidden = self.depthwise(value) * torch.sigmoid(gate)
        return signal + self.output(F.gelu(self.norm(hidden)))


class TCNAutoencoder(nn.Module):
    """输入和输出均为 [batch, ACC1/ACC2/ACC3, 2048]，每窗一秒。"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(3, 64, 64, stride=32, padding=16), nn.GELU())
        self.temporal = nn.Sequential(*[
            _TCNBlock(64, 76, dilation) for dilation in [1, 2, 4, 8, 16, 32, 1, 2]
        ])
        self.decoder = nn.ConvTranspose1d(64, 3, 64, stride=32, padding=16)

    def forward(self, signal):
        return self.decoder(self.temporal(self.encoder(signal)))
