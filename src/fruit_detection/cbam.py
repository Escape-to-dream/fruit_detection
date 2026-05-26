import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1))) * x


class CBAM(nn.Module):
    def __init__(self, channels=None, reduction=16, kernel_size=7):
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_attention = None
        self.spatial_attention = None
        if channels is not None:
            self._init_layers(channels)

    def _init_layers(self, channels):
        self.channel_attention = ChannelAttention(channels, self.reduction)
        self.spatial_attention = SpatialAttention(self.kernel_size)

    def forward(self, x):
        if self.channel_attention is None:
            self._init_layers(x.shape[1])
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x