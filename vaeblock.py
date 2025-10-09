import numpy as np
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as sn



def normalization(channels, norm_type="group", num_groups=32):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    elif (not norm_type) or (norm_type.tolower() == 'none'):
        return nn.Identity()
    else:
        raise NotImplementedError(norm)


def activation(act_type="swish"):
    if act_type == "swish":
        return nn.SiLU()
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "tanh":
        return nn.Tanh()
    elif not act_type:
        return nn.Identity()
    else:
        raise NotImplementedError(act_type)


class ResBlock3D(nn.Module):
    def __init__(
            self, in_channels, out_channels, resample=None,
            resample_factor=(1, 1, 1), kernel_size=(3, 3, 3),
            act='swish', norm='group', norm_kwargs=None,
            spectral_norm=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

        padding = tuple(k // 2 for k in kernel_size)
        if resample == "down":
            self.resample = nn.AvgPool3d(resample_factor, ceil_mode=True)
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=resample_factor, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
        elif resample == "up":
            self.resample = nn.Upsample(
                scale_factor=resample_factor, mode='trilinear')
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels,
                                            kernel_size=kernel_size, padding=padding)
            output_padding = tuple(
                2 * p + s - k for (p, s, k) in zip(padding, resample_factor, kernel_size)
            )
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels,
                                            kernel_size=kernel_size, stride=resample_factor,
                                            padding=padding, output_padding=output_padding)
        else:
            self.resample = nn.Identity()
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding)

        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm1 = normalization(in_channels, norm_type=norm, **norm_kwargs)
        self.norm2 = normalization(out_channels, norm_type=norm, **norm_kwargs)
        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if not isinstance(self.proj, nn.Identity):
                self.proj = sn(self.proj)

    def forward(self, x):
        x_in = self.resample(self.proj(x))
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + x_in


class SimpleConvEncoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64):
        sequence = []
        channels = np.hstack([
            in_dim, 
            (8**np.arange(1,levels+1)).clip(min=min_ch)
        ])
        
        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i+1])
            res_kernel_size = (3,3,3) if i == 0 else (1,3,3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            downsample = nn.Conv3d(out_channels, out_channels,
                kernel_size=(2,2,2), stride=(2,2,2))
            sequence.append(downsample)
            in_channels = out_channels

        super().__init__(*sequence)


class SimpleConvDecoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64):
        sequence = []
        channels = np.hstack([
            in_dim, 
            (8**np.arange(1,levels+1)).clip(min=min_ch)
        ])

        for i in reversed(list(range(levels))):
            in_channels = int(channels[i+1])
            out_channels = int(channels[i])
            upsample = nn.ConvTranspose3d(in_channels, in_channels, 
                    kernel_size=(2,2,2), stride=(2,2,2))
            sequence.append(upsample)
            res_kernel_size = (3,3,3) if (i == 0) else (1,3,3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            in_channels = out_channels

        super().__init__(*sequence)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x