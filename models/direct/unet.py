"""Simple UNet-style architecture for diffusion forecasting."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int, num_groups: int) -> nn.GroupNorm:
    groups = min(num_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups or 1, num_channels)


class _DoubleConv(nn.Module):
    """(Conv → GroupNorm → SiLU) × 2."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            _make_group_norm(out_channels, norm_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            _make_group_norm(out_channels, norm_groups),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels, out_channels, norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class _UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm_groups: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_channels + skip_channels, out_channels, norm_groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # Handle potential off-by-one mismatches caused by odd-sized inputs during
        # pooling/upsampling cycles. We follow the common UNet practice of
        # symmetrically padding or cropping the upsampled tensor to align with the
        # skip connection feature map before concatenation.
        diff_h = skip.shape[-2] - x.shape[-2]
        diff_w = skip.shape[-1] - x.shape[-1]

        if diff_h > 0 or diff_w > 0:
            pad_top = diff_h // 2 if diff_h > 0 else 0
            pad_bottom = diff_h - pad_top if diff_h > 0 else 0
            pad_left = diff_w // 2 if diff_w > 0 else 0
            pad_right = diff_w - pad_left if diff_w > 0 else 0
            if pad_top or pad_bottom or pad_left or pad_right:
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        if x.shape[-2] > skip.shape[-2]:
            crop_top = (x.shape[-2] - skip.shape[-2]) // 2
            crop_bottom = crop_top + skip.shape[-2]
            x = x[:, :, crop_top:crop_bottom, :]

        if x.shape[-1] > skip.shape[-1]:
            crop_left = (x.shape[-1] - skip.shape[-1]) // 2
            crop_right = crop_left + skip.shape[-1]
            x = x[:, :, :, crop_left:crop_right]

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Simple3DUNet(nn.Module):
    """A light-weight UNet that operates on flattened spatio-temporal maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers must contain at least one element")

        widths = [base_channels * m for m in channel_multipliers]
        self.stem = _DoubleConv(in_channels, widths[0], norm_groups)

        self.down_blocks = nn.ModuleList()
        for idx in range(len(widths) - 1):
            self.down_blocks.append(_DownBlock(widths[idx], widths[idx + 1], norm_groups))

        bottleneck_channels = widths[-1] * 2
        self.bottleneck = _DoubleConv(widths[-1], bottleneck_channels, norm_groups)

        self.up_blocks = nn.ModuleList()
        up_in_channels = bottleneck_channels
        for skip_channels in reversed(widths):
            self.up_blocks.append(_UpBlock(up_in_channels, skip_channels, skip_channels, norm_groups))
            up_in_channels = skip_channels

        self.head = nn.Conv2d(up_in_channels, out_channels, kernel_size=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        skips = [features]
        for down in self.down_blocks:
            features = down(features)
            skips.append(features)

        features = self.bottleneck(features)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            features = up(features, skip)

        return self.head(features)


__all__ = ["Simple3DUNet"]
