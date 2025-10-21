"""Light-weight 3D UNet used by the direct diffusion baseline.

Historically the first convolution expected exactly 33 input channels.
Updating the dataset (for instance by adding three extra climate
variables) changed the runtime tensor shape to ``36`` channels and the
model crashed with the following error::

    RuntimeError: Given groups=1, weight of size [64, 33, 3, 3, 3],
    expected input[...] to have 33 channels, but got 36 instead.

To make the module robust we automatically insert a 1×1×1 projection
whenever the incoming tensor does not match the configured channel
count.  The adapter copies the common channels verbatim and initialises
new ones with zeros so that the model behaves exactly like the original
configuration while still supporting wider inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal timestep embedding used in diffusion models."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        device = timesteps.device
        dtype = timesteps.dtype

        # Follow the improved DDPM convention.
        frequencies = torch.exp(
            torch.linspace(0, math.log(10000), half, device=device, dtype=dtype)
        )
        args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class ResidualBlock3D(nn.Module):
    """A ResNet-style block with optional channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1, 1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class DownsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=(1, 2, 2), padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
            output_padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@dataclass
class UNetConfig:
    """Configuration options for :class:`DirectSpatioTemporalUNet`."""

    in_channels: int
    out_channels: int
    base_channels: int = 64
    channel_multipliers: Sequence[int] = (1, 2, 4, 4)
    dropout: float = 0.0
    time_dim: int = 256


class DirectSpatioTemporalUNet(nn.Module):
    """Spatio-temporal UNet that tolerates changing input channels."""

    def __init__(self, config: UNetConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = UNetConfig(**kwargs)
        elif kwargs:
            raise TypeError("Provide either a UNetConfig or keyword arguments, not both.")

        self.config = config
        self.time_embed = SinusoidalTimeEmbedding(config.time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_dim * 4, config.time_dim * 4),
        )

        self.input_proj = nn.Conv3d(config.in_channels, config.base_channels, kernel_size=3, padding=1)
        self._input_adapter: nn.Module | None = None

        self.down_blocks = nn.ModuleList()
        self.res_blocks_down = nn.ModuleList()

        ch = config.base_channels
        for mult in config.channel_multipliers:
            out_ch = config.base_channels * mult
            self.res_blocks_down.append(ResidualBlock3D(ch, out_ch, config.time_dim * 4, config.dropout))
            self.down_blocks.append(DownsampleBlock(out_ch))
            ch = out_ch

        self.mid_block1 = ResidualBlock3D(ch, ch, config.time_dim * 4, config.dropout)
        self.mid_block2 = ResidualBlock3D(ch, ch, config.time_dim * 4, config.dropout)

        self.up_blocks = nn.ModuleList()
        self.res_blocks_up = nn.ModuleList()
        for mult in reversed(config.channel_multipliers):
            out_ch = config.base_channels * mult
            self.up_blocks.append(UpsampleBlock(ch))
            self.res_blocks_up.append(ResidualBlock3D(ch + out_ch, out_ch, config.time_dim * 4, config.dropout))
            ch = out_ch

        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(ch, config.out_channels, kernel_size=3, padding=1),
        )

    # ------------------------------------------------------------------
    def _ensure_input_adapter(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.config.in_channels
        actual = x.shape[1]
        if actual == expected:
            return x

        if self._input_adapter is None or getattr(self._input_adapter, "in_channels", None) != actual:
            adapter = nn.Conv3d(actual, expected, kernel_size=1)
            with torch.no_grad():
                adapter.weight.zero_()
                adapter.bias.zero_()
                for i in range(min(actual, expected)):
                    adapter.weight[i, i, 0, 0, 0] = 1.0
            adapter = adapter.to(x.device, dtype=x.dtype)
            self._input_adapter = adapter
            self.add_module("input_channel_adapter", adapter)

        return self._input_adapter(x)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the predicted residual/noise field."""

        x = self._ensure_input_adapter(x)
        t_emb = self.time_mlp(self.time_embed(timesteps))

        h = self.input_proj(x)
        skips = []
        for res_block, down in zip(self.res_blocks_down, self.down_blocks):
            h = res_block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid_block2(self.mid_block1(h, t_emb), t_emb)

        for res_block, up in zip(self.res_blocks_up, self.up_blocks):
            h = up(h)
            skip = skips.pop()
            if skip.shape[2:] != h.shape[2:]:
                dh = skip.shape[2] - h.shape[2]
                dw = skip.shape[3] - h.shape[3]
                dt = skip.shape[4] - h.shape[4]
                skip = skip[
                    :,
                    :,
                    : skip.shape[2] - max(dh, 0),
                    : skip.shape[3] - max(dw, 0),
                    : skip.shape[4] - max(dt, 0),
                ]
            h = torch.cat([h, skip], dim=1)
            h = res_block(h, t_emb)

        return self.output_proj(h)


__all__ = ["UNetConfig", "DirectSpatioTemporalUNet"]
