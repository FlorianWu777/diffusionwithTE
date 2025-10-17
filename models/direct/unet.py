import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.utils import timestep_embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_channels)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        t = self.time_proj(t_emb).type_as(h)
        while len(t.shape) < len(h.shape):
            t = t[..., None]
        h = h + t
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, downsample: bool):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.res1(x, t_emb)
        h = self.res2(h, t_emb)
        skip = h
        if self.downsample is not None:
            h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int, upsample: bool):
        super().__init__()
        self.upsample = None
        if upsample:
            self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.res1 = ResidualBlock(in_channels + skip_channels, out_channels, time_dim)
        self.res2 = ResidualBlock(out_channels, out_channels, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor):
        if self.upsample is not None:
            x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class Simple3DUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 3,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth

        self.time_embed_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.input_proj = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)

        downs = []
        channels = base_channels
        skip_channels = []
        for level in range(depth):
            out_ch = base_channels * (2 ** level)
            downs.append(DownBlock(channels, out_ch, self.time_embed_dim, downsample=(level != depth - 1)))
            skip_channels.append(out_ch)
            channels = out_ch
        self.downs = nn.ModuleList(downs)

        self.mid = ResidualBlock(channels, channels, self.time_embed_dim)

        ups = []
        for level in reversed(range(depth)):
            skip_ch = skip_channels[level]
            out_ch = base_channels * (2 ** level)
            ups.append(UpBlock(channels, skip_ch, out_ch, self.time_embed_dim, upsample=(level != 0)))
            channels = out_ch
        self.ups = nn.ModuleList(ups)

        self.output_norm = nn.GroupNorm(8, channels)
        self.output_act = nn.SiLU()
        self.output_proj = nn.Conv3d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor = None, context=None) -> torch.Tensor:
        if timesteps is None:
            raise ValueError("timesteps tensor is required")
        temb = timestep_embedding(timesteps, self.base_channels)
        temb = self.time_mlp(temb)

        h = self.input_proj(x)
        skips = []
        for down in self.downs:
            h, skip = down(h, temb)
            skips.append(skip)

        h = self.mid(h, temb)

        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, temb)

        h = self.output_act(self.output_norm(h))
        return self.output_proj(h)
