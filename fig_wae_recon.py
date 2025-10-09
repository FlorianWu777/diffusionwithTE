# wae_recon_plot_siconca_blues.py
# 使用 matplotlib 预设 Blues_r colormap

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from trainvqvae import ClimateForecastDataset  # 你的数据类

# ===== 模型模块省略，保持和你训练一致 =====
# （ConvEncoder3D, ConvDecoder3D, AutoencoderWAE 同之前）
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from torch.utils.data import DataLoader, Subset
from trainvqvae import ClimateForecastDataset  # 你的数据类

from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

# ----------------- 复制必要的模型模块（与训练保持一致） -----------------
def safe_groupnorm(c: int, g: int = 8) -> nn.GroupNorm:
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)

class ResidualBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm1 = safe_groupnorm(ch)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1)
        self.norm2 = safe_groupnorm(ch)
        self.act   = nn.GELU()
    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + x)

class PatchAttention3D(nn.Module):
    def __init__(self, ch: int, patch: int = 2, heads: int = 4):
        super().__init__()
        self.p   = patch
        self.norm = safe_groupnorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.proj = nn.Linear(ch, ch)
    def forward(self, x):  # [B,C,T,H,W]
        b, c, t, h, w = x.shape
        p = self.p
        assert h % p == 0 and w % p == 0, "H/W must be divisible by patch size"
        x = self.norm(x).view(b, c, t, h//p, p, w//p, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1).contiguous()  # [B T H_ W_ p p C]
        seq = p * p
        x = x.view(b * t * (h//p) * (w//p), seq, c)
        attn_out, _ = self.attn(x, x, x)
        x = self.proj(attn_out) + x
        x = x.view(b, t, h//p, w//p, p, p, c).permute(0, 6, 1, 2, 4, 3, 5)
        return x.contiguous().view(b, c, t, h, w)

class ConvEncoder3D(nn.Module):
    def __init__(self, in_ch: int = 4, fine_ch: int = 32, coarse_ch: Optional[int] = None, patch: int = 2):
        super().__init__()
        coarse_ch = coarse_ch or fine_ch
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, stride=(1,2,2), padding=1), nn.GELU(), ResidualBlock3D(8),
            nn.Conv3d(8, 16, 3, stride=(1,2,2), padding=1),  nn.GELU(), ResidualBlock3D(16),
            nn.Conv3d(16, 32,3, stride=(1,2,2), padding=1),  nn.GELU(), ResidualBlock3D(32),
        )
        self.mom_fine = nn.Conv3d(32, 2*fine_ch, 1)
        self.attn   = PatchAttention3D(32, patch)
        self.down   = nn.Conv3d(32, coarse_ch, 3, stride=(1,2,2), padding=1)
        self.mom_coarse = nn.Conv3d(coarse_ch, 2*coarse_ch, 1)
    def forward(self, x):
        h = self.net(x)
        h = self.down(self.attn(h))
        mu_c, log_c = torch.chunk(self.mom_coarse(h), 2, 1)
        return mu_c, log_c

class ConvDecoder3D(nn.Module):
    def __init__(self, fine_ch: int = 32, coarse_ch: Optional[int] = None, out_ch: int = 4, patch: int = 2):
        super().__init__()
        coarse_ch = coarse_ch or fine_ch
        self.up   = nn.ConvTranspose3d(coarse_ch, coarse_ch, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.to_fuse = nn.Conv3d(coarse_ch, coarse_ch, 1)
        self.attn = PatchAttention3D(coarse_ch, patch)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(coarse_ch, 64, (1,4,4), stride=(1,2,2), padding=(0,1,1)), nn.GELU(), ResidualBlock3D(64),
            nn.ConvTranspose3d(64, 32, (1,4,4), stride=(1,2,2), padding=(0,1,1)),  nn.GELU(), ResidualBlock3D(32),
            nn.ConvTranspose3d(32, 16, (1,4,4), stride=(1,2,2), padding=(0,1,1)),  nn.GELU(), ResidualBlock3D(16),
            nn.Conv3d(16, out_ch, 3, padding=1)
        )
    def forward(self, z_coarse):
        z = self.up(z_coarse)
        z = F.gelu(self.to_fuse(z))
        z = self.attn(z)
        return self.dec(z)

from wae_train import AutoencoderWAE, ConvEncoder3D, ConvDecoder3D

# ----------------- 可选：反归一化（按需修改） -----------------
def denorm_siconca(x):
    # 如已标准化/归一化，请在此还原；默认假设已是 [0,1]
    return x

# ----------------- 可选：反归一化 -----------------
def denorm_siconca(x):
    # 如已标准化/归一化，请在此还原；默认假设已是 [0,1]
    return x

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CKPT = "/data/wuhaotian/diffusionDemo/model/wae_finetune_best.pt"
    OUT_DIR = './fig_wae_recon_blues'
    os.makedirs(OUT_DIR, exist_ok=True)

    DATASET_ROOT = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = variables[:]
    SICONCA_INDEX = 3

    reanal_dataset = ClimateForecastDataset(DATASET_ROOT, variables, target_var,
                                            input_seq_len=12, output_seq_len=1, mode='obs')
    total_len = len(reanal_dataset)
    test_indices  = list(range(total_len - 120, total_len))
    test_set = Subset(reanal_dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # 模型（保持和训练一致）
    in_channels = len(variables)
    latent_channels = 32
    encoder = ConvEncoder3D(in_ch=in_channels, fine_ch=latent_channels).to(DEVICE)
    decoder = ConvDecoder3D(fine_ch=latent_channels, out_ch=in_channels).to(DEVICE)
    wae = AutoencoderWAE(encoder=encoder, decoder=decoder).to(DEVICE)

    # 加载权重
    sd = torch.load(CKPT, map_location=DEVICE)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
        new_sd = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith('module.') else k
            new_sd[nk] = v
        sd = new_sd
    wae.load_state_dict(sd, strict=True)
    wae.eval()

    # ========= 逐样本画图 =========
    MAX_FIGS = 12
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= MAX_FIGS:
                break
            x = batch[0].to(DEVICE)
            recon, _ = wae(x)
#            print(f"[Case {i}] latent z shape: {_.shape}")

            truth_sic = x[0, SICONCA_INDEX, -1].detach().cpu().numpy()
            recon_sic = recon[0, SICONCA_INDEX, -1].detach().cpu().numpy()

            truth_sic = denorm_siconca(truth_sic)
            recon_sic = denorm_siconca(recon_sic)
            abs_err   = np.abs(recon_sic - truth_sic)

            vmin, vmax = 0.0, 1.0
            emax = np.nanpercentile(abs_err, 98)
            emax = max(emax, 0.05)

            fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), dpi=150, constrained_layout=True)

            im0 = axs[0].imshow(truth_sic, origin='lower', cmap='Blues_r', vmin=vmin, vmax=vmax)
            axs[0].set_title('Truth siconca')
            plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            im1 = axs[1].imshow(recon_sic, origin='lower', cmap='Blues_r', vmin=vmin, vmax=vmax)
            axs[1].set_title('Reconstruction')
            plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            im2 = axs[2].imshow(abs_err, origin='lower', cmap='Blues_r', vmin=0.0, vmax=emax)
            axs[2].set_title('|Recon − Truth|')
            plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])

            out_path = os.path.join(OUT_DIR, f'wae_recon_siconca_case{i:03d}.png')
            plt.savefig(out_path)
            plt.close(fig)
            print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
