from __future__ import annotations
import torch
from torch import nn
import pytorch_lightning as pl
import os
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from dataloader import ClimateForecastDataset
from trainvqvae import ClimateForecastDataset
import torch.nn.functional as F
from typing import Optional, Iterable, Tuple
import os
os.environ["PYTHONUNBUFFERED"] = "1"
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Autoencoder WAE (3‑D Climate, DDP‑friendly) — 2025‑09‑13

This script adapts a VAE to a Wasserstein Auto‑Encoder (WAE‑MMD) and
adds optional cross‑domain alignment (CMIP ↔ Reanalysis) and optional
all_gather‑based global MMD for multi‑GPU stability.

Pipeline:
1) Pretrain on CMIP (reconstruction + latent MMD to N(0,I)).
2) Optional alignment stage: jointly sample CMIP & Reanalysis batches,
   add cross‑domain MMD(z_cmip, z_reanal).
3) Finetune on Reanalysis only.

Latent Diffusion hook: the learned encoder/decoder provide a calibrated
latent space z ~ N(0,I). Train a DDPM in latent space with an
attention‑U‑Net denoiser; at inference decode the sampled z.

Notes:
- Replaces KL with MMD (RBF multi‑bandwidth). See Tolstikhin et al.,
  "Wasserstein Auto‑Encoders" (ICLR 2018) and Gretton et al.,
  "A Kernel Two‑Sample Test" (JMLR 2012).
- Optionally switch to sliced‑Wasserstein or Sinkhorn OT if desired.

Edit the `DATASET_ROOT` and dataset constructors to match your setup.
"""

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

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
    """Self‑attention in p×p spatial patches for every time frame."""
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
        x = x.permute(0, 2, 3, 5, 4, 6, 1).contiguous()  # → [B T H_ W_ p p C]
        seq = p * p
        x = x.view(b * t * (h//p) * (w//p), seq, c)
        attn_out, _ = self.attn(x, x, x)
        x = self.proj(attn_out) + x
        x = x.view(b, t, h//p, w//p, p, p, c).permute(0, 6, 1, 2, 4, 3, 5)
        return x.contiguous().view(b, c, t, h, w)

# -----------------------------------------------------------------------------
# Encoder / Decoder (coarse latent only, matching your current setup)
# -----------------------------------------------------------------------------

class ConvEncoder3D(nn.Module):
    def __init__(self, in_ch: int = 4, fine_ch: int = 32, coarse_ch: Optional[int] = None, patch: int = 2):
        super().__init__()
        coarse_ch = coarse_ch or fine_ch
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, stride=(1,2,2), padding=1), nn.GELU(), ResidualBlock3D(8),
            nn.Conv3d(8, 16, 3, stride=(1,2,2), padding=1),  nn.GELU(), ResidualBlock3D(16),
            nn.Conv3d(16, 32,3, stride=(1,2,2), padding=1),  nn.GELU(), ResidualBlock3D(32),
        )  # output H/8
        self.mom_fine = nn.Conv3d(32, 2*fine_ch, 1)  # kept for compatibility (not used)
        self.attn   = PatchAttention3D(32, patch)
        self.down   = nn.Conv3d(32, coarse_ch, 3, stride=(1,2,2), padding=1)  # H/16
        self.mom_coarse = nn.Conv3d(coarse_ch, 2*coarse_ch, 1)

    def forward(self, x):
        h = self.net(x)
        # mu_f, log_f = torch.chunk(self.mom_fine(h), 2, 1)  # fine unused
        h = self.down(self.attn(h))
        mu_c, log_c = torch.chunk(self.mom_coarse(h), 2, 1)
        return mu_c, log_c
        
class Up3D(nn.Module):
    def __init__(self, ch_in, ch_out, scale=(1,2,2)):
        super().__init__()
        self.scale=scale
        self.conv = nn.Conv3d(ch_in, ch_out, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return F.gelu(self.conv(x))
class ConvDecoder3D(nn.Module):
    def __init__(self, fine_ch: int = 32, coarse_ch: Optional[int] = None, out_ch: int = 4, patch: int = 2):
        super().__init__()
        coarse_ch = coarse_ch or fine_ch
        #self.up   = nn.ConvTranspose3d(coarse_ch, coarse_ch, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.up = Up3D(coarse_ch,coarse_ch)
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



# -----------------------------------------------------------------------------
# WAE pieces: MMD utilities (RBF multi‑bandwidth, with optional median trick)
# -----------------------------------------------------------------------------

def _flatten_feat(t: torch.Tensor) -> torch.Tensor:
    return t.view(t.shape[0], -1)

@torch.no_grad()
def _median_pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    x = _flatten_feat(x)
    dists = torch.cdist(x, x, p=2)
    med = torch.median(dists[dists>0])
    if not torch.isfinite(med):
        med = torch.tensor(1.0, device=x.device)
    return med

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: Optional[Iterable[float]] = None,
            use_median: bool = True) -> torch.Tensor:
    """MMD^2 with multi‑bandwidth RBF kernel (biased estimate).
    If sigmas=None and use_median=True, use median heuristic to set a scale and
    make a small geometric ladder around it.
    """
    x = _flatten_feat(x)
    y = _flatten_feat(y)

    if sigmas is None:
        if use_median:
            with torch.no_grad():
                scale = _median_pairwise_distance(torch.cat([x, y], dim=0)).clamp_min(1e-3)
            # geometric ladder around median
            sigmas = [float(scale * (2.0 ** k)) for k in (-2, -1, 0, 1, 2)]
        else:
            sigmas = [0.5, 1., 2., 4., 8.]

    x2 = (x**2).sum(dim=1, keepdim=True)
    y2 = (y**2).sum(dim=1, keepdim=True)
    xy = x @ y.t()
    xx = x @ x.t()
    yy = y @ y.t()

    dxx = x2 + x2.t() - 2*xx
    dyy = y2 + y2.t() - 2*yy
    dxy = x2 + y2.t() - 2*xy

    mmd2 = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * float(s) ** 2)
        k_xx = torch.exp(-gamma * dxx)
        k_yy = torch.exp(-gamma * dyy)
        k_xy = torch.exp(-gamma * dxy)
        mmd2 += k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return mmd2 / len(sigmas)

# -----------------------------------------------------------------------------
# Optional: global all_gather to stabilize MMD on multi‑GPU
# -----------------------------------------------------------------------------

def all_gather_concat(t: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return t
    world = dist.get_world_size()
    if world == 1:
        return t
    tensors = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(tensors, t.contiguous())
    return torch.cat(tensors, dim=0)
def moment_reg(z):
    # z: [B,C,T',H',W'] → 按通道做一阶/二阶矩吻合
    zf = z.view(z.shape[0], z.shape[1], -1)
    m  = zf.mean(dim=-1)              # [B,C]
    v  = zf.var (dim=-1, unbiased=False)
    loss_m = (m**2).mean()            # 目标均值→0
    loss_v = ((v-1.0)**2).mean()      # 目标方差→1
    return 0.1*loss_m + 0.1*loss_v

# -----------------------------------------------------------------------------
# AutoencoderWAE (coarse latent only)
# -----------------------------------------------------------------------------

class AutoencoderWAE(pl.LightningModule):
    # 加到 __init__ 的参数里（可选）
    def __init__(self, encoder, decoder, lambda_mmd=10.0, add_enc_noise=False, lr=1e-3,
                 use_global_mmd=False, sic_idx: int = 1, use_edge_hf_for_sic: bool = True,  lambda_mom: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lambda_mmd = lambda_mmd
        self.add_enc_noise = add_enc_noise
        self.lr = lr
        self.use_global_mmd = use_global_mmd
        self.sic_idx = sic_idx                 # siconca/abs 在输入中的通道索引
        self.use_edge_hf_for_sic = use_edge_hf_for_sic
        self.lambda_mom = lambda_mom
        self.hidden_width = 32
       # self.scaling_factor = scaling_factor

    # 统一的重建损失：siconca 用 recon_loss，其它通道用 L1
    def recon_loss_channels(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        x, recon: [B, C, T, H, W]
        对 sic_idx 通道使用你定义的 recon_loss（边缘+高频），
        其余通道使用 L1。
        """
        B, C, T, H, W = x.shape
        assert x.shape == recon.shape

        # 其余通道（不含 sic_idx）
        if C > 1:
            keep = [i for i in range(C) if i != self.sic_idx]
            l1_others = F.l1_loss(recon[:, keep, ...], x[:, keep, ...])
        else:
            l1_others = torch.tensor(0., device=x.device, dtype=x.dtype)

        # siconca 通道（形状变成 [B*T, H, W] 再喂 recon_loss）
        if (0 <= self.sic_idx < C) and self.use_edge_hf_for_sic:
            y     = x[:, self.sic_idx:self.sic_idx+1, ...].squeeze(1).reshape(-1, H, W)
            y_hat = recon[:, self.sic_idx:self.sic_idx+1, ...].squeeze(1).reshape(-1, H, W)
            sic_loss = recon_loss(y, y_hat)   # ← 你上面定义的函数
        else:
            sic_loss = F.l1_loss(recon, x)    # 兜底

        return l1_others + sic_loss

    @torch.no_grad()
    def _sample_prior(self, like: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(like)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)  # (mu_c, log_c)
        mu_c , log_c = out


        if isinstance(out, (tuple, list)) and len(out) == 2:
            mu_c, log_c = out
            z = mu_c
            if self.training and self.add_enc_noise:
                std = (0.5 * log_c).exp()
                z = mu_c + torch.randn_like(std) * std * 0.1
            return z,mu_c,log_c
       
        return z,mu_c, log_c

    def decode(self, z_scaled: torch.Tensor) -> torch.Tensor:
        #return self.decoder(z_scaled / self.scaling_factor)
        return self.decoder(z_scaled)

    def forward(self, x: torch.Tensor):
        z,mu_c, log_c = self.encode(x)
        #recon = self.decode(z * self.scaling_factor)
        recon = self.decode(z)
        return recon, (mu_c, log_c)

    def _loss_core(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon, z = self(x)
        rec = self.recon_loss_channels(x, recon)
    
        # MMD 可用全局拼接（无梯度到其它卡也没关系）
        z_eval = all_gather_concat(z) if self.use_global_mmd else z
        z_prior = torch.randn_like(z_eval)
        mmd = mmd_rbf(z_eval, z_prior)
    
        # ← NEW: 本机 z 做矩匹配（可回传梯度）
        mom = moment_reg(z)
    
        total = rec + self.lambda_mmd * mmd + self.lambda_mom * mom
        return total, rec, mmd
        
    def training_step(self, batch, _):
        x, = batch
        tot, rec, mmd = self._loss_core(x)
        self.log_dict({"train_loss": tot, "train_rec": rec, "train_mmd": mmd})
        return tot

    def validation_step(self, batch, _):
        x, = batch
        recon, z = self(x)

        # 只在第一批、主进程打印一次
        if batch_idx == 0 and self.global_rank == 0:
            self.print(f"[VAL DEBUG] x={tuple(x.shape)}  z={tuple(z.shape)}  recon={tuple(recon.shape)}")

        tot, rec, mmd = self._loss_core(x)
        self.log_dict({"val_loss": tot, "val_rec": rec, "val_mmd": mmd}, prog_bar=True)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=1e-3)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.25, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_rec"}}

# -----------------------------------------------------------------------------
# Cross‑domain alignment training utilities
# -----------------------------------------------------------------------------

def evaluate_autoencoder(loader: DataLoader, model: nn.Module, device: torch.device) -> Tuple[float,float,float]:
    model.eval()
    total = rec_total = mmd_total = 0.0
    with torch.no_grad():
        for batch in loader:
            x, = (b.to(device) for b in batch[:1])
            recon, z = model(x)
            rec = model.recon_loss_channels(x, recon) 
            if isinstance(model, AutoencoderWAE) and model.use_global_mmd:
                z_eval = all_gather_concat(z)
            else:
                z_eval = z
            z_prior = torch.randn_like(z_eval)
            mmd = mmd_rbf(z_eval, z_prior).item()
            total += rec + model.lambda_mmd * mmd
            rec_total += rec
            mmd_total += mmd
    n = len(loader)
    return total / n, rec_total / n, mmd_total / n

@torch.no_grad()
def _next(loader_iter, fallback_iter):
    try:
        return next(loader_iter)
    except StopIteration:
        return next(fallback_iter)

def align_two_domains_epoch(model: AutoencoderWAE, opt: torch.optim.Optimizer, device,
                            cmip_loader: DataLoader, reanal_loader: DataLoader,
                            lambda_cross: float = 2.0, desc: str = "Align"):
    model.train()
    cmip_it = iter(cmip_loader)
    rean_it = iter(reanal_loader)
    cmip_backup = iter(cmip_loader)
    rean_backup = iter(reanal_loader)

    steps = min(len(cmip_loader), len(reanal_loader))
    prog = tqdm(range(steps), desc=desc, disable=not _is_main())

    for _ in prog:
        x_c, = (b.to(device) for b in _next(cmip_it, cmip_backup)[:1])
        x_r, = (b.to(device) for b in _next(rean_it, rean_backup)[:1])

        # core losses
        recon_c, z_c = model(x_c)
        recon_r, z_r = model(x_r)
        rec_c = model.recon_loss_channels(x_c, recon_c)   # ← 新
        rec_r = model.recon_loss_channels(x_r, recon_r)   # ← 新
        rec   = 0.5 * (rec_c + rec_r)
        # global concat for stability (optional)
        z_eval_c = all_gather_concat(z_c) if model.use_global_mmd else z_c
        z_eval_r = all_gather_concat(z_r) if model.use_global_mmd else z_r

        mmd_c = mmd_rbf(z_eval_c, torch.randn_like(z_eval_c))
        mmd_r = mmd_rbf(z_eval_r, torch.randn_like(z_eval_r))
        mmd_cross = mmd_rbf(z_eval_c, z_eval_r)

        mom_c = moment_reg(z_c)                     # ← NEW
        mom_r = moment_reg(z_r)                     # ← NEW
        mom    = 0.5 * (mom_c + mom_r)              # ← NEW
        
        tot = rec + model.lambda_mmd * (mmd_c + mmd_r) * 0.5 \
                + lambda_cross * mmd_cross \
                + model.lambda_mom * mom            # ← NEW

        opt.zero_grad(set_to_none=True)
        tot.backward()
        opt.step()

        if _is_main():
            prog.set_postfix(rec=float(rec.detach()), mmd=float(((mmd_c+mmd_r)/2).detach()), cross=float(mmd_cross.detach()))

# -----------------------------------------------------------------------------
# Training loops (pretrain / align / finetune)
# -----------------------------------------------------------------------------

def _is_main() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def train_autoencoder_loop(train_loader: DataLoader, val_loader: DataLoader, model: AutoencoderWAE,
                           optimizer, device, writer: Optional[SummaryWriter], epochs: int,
                           desc: str, save_path: str, local_rank: int = 0):
    best_val = float("inf")
    for epoch in range(epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        running = {"tot": 0.0, "rec": 0.0, "mmd": 0.0}
        prog = tqdm(train_loader, desc=f"[{desc}] Epoch {epoch+1}/{epochs}", disable=(local_rank != 0))
        for x, *_ in prog:
            x = x.to(device)
            recon, z = model(x)
            rec = model.recon_loss_channels(x, recon)         # ← 替换

            z_eval = all_gather_concat(z) if model.use_global_mmd else z
            mmd = mmd_rbf(z_eval, torch.randn_like(z_eval))
            mom = moment_reg(z)                                   # ← NEW
            tot = rec + model.lambda_mmd * mmd + model.lambda_mom * mom   # ← NEW
            optimizer.zero_grad(set_to_none=True)
            tot.backward()
            optimizer.step()
            running["tot"] += float(tot.detach())
            running["rec"] += float(rec.detach())
            running["mmd"] += float(mmd.detach())
        if writer and local_rank == 0:
            n = len(train_loader)
            writer.add_scalars(f"{desc}/train", {k: v/n for k, v in running.items()}, epoch)
        if local_rank == 0:
            val_tot, val_rec, val_mmd = evaluate_autoencoder(val_loader, model, device)
            if writer:
                writer.add_scalars(f"{desc}/val", {"tot": val_tot, "rec": val_rec, "mmd": val_mmd}, epoch)
            if val_tot < best_val:
                best_val = val_tot
                torch.save(model.state_dict(), save_path)
                print(f"✅  Best model saved @ epoch {epoch+1}  val_tot={val_tot:.4f}")
import torch, torch.nn.functional as F

def sobel_grad(x):
    # 对每帧做 2D Sobel（只示例 H,W，T 可展平或逐帧）
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    x = x.reshape(-1,1,x.shape[-2],x.shape[-1])
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    g  = torch.sqrt(gx**2 + gy**2 + 1e-12)
    return g.reshape(-1, *g.shape[-2:])  # 按需还原

def fourier_hf_loss(y, y_hat, alpha=1.0):
    # 频域差异，对高频给更大权重
    def fft2_mag(z):
        Z = torch.fft.rfft2(z, norm="ortho")
        return torch.abs(Z)
    Y, Yh = fft2_mag(y), fft2_mag(y_hat)
    # 构造径向权重 (k/r_max)^alpha
    H, W = y.shape[-2], y.shape[-1]
    ky = torch.fft.rfftfreq(H, d=1.0).to(y.device)
    kx = torch.fft.rfftfreq(W, d=1.0).to(y.device)
    wy = ky[:, None]; wx = kx[None, :]
    r  = torch.sqrt(wy**2 + wx**2)
    w  = (r / (r.max() + 1e-12))**alpha
    return ((w * (Y - Yh).abs()).mean())

def edge_weight(y, thr=0.15, band=0.05):
    # 对靠近等值线的像素放大权重
    w = torch.exp(-((y - thr).abs() / band))
    return 1.0 + 3.0 * w   # 最高 4 倍权重，可调

def recon_loss(y, y_hat):
    w = edge_weight(y)
    l1 = (w * (y_hat - y).abs()).mean()
    grad = (sobel_grad(y_hat) - sobel_grad(y)).abs().mean()
    #hf = fourier_hf_loss(y, y_hat, alpha=1.0)
    return l1 + 0.2*grad #+ 0.05*hf   # 建议起始权重
# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    torch.backends.cudnn.benchmark = False 
    # ====== DDP init ======
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_ddp = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ====== Data ======
    DATASET_ROOT = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']

    cmip_names = [
        'EC-Earth3/r2i1p1f1','EC-Earth3/r7i1p1f1','EC-Earth3/r10i1p1f1','EC-Earth3/r12i1p1f1','EC-Earth3/r14i1p1f1',
        'MRI-ESM2-0/r1i1p1f1','MRI-ESM2-0/r2i1p1f1','MRI-ESM2-0/r3i1p1f1','MRI-ESM2-0/r4i1p1f1','MRI-ESM2-0/r5i1p1f1']

    cmip_dataset = ClimateForecastDataset(DATASET_ROOT, variables, target_var, input_seq_len=12, output_seq_len=12, mode='transfer')
    cmip_sampler = DistributedSampler(cmip_dataset, shuffle=True, drop_last=True) if use_ddp else None
    cmip_loader = DataLoader(cmip_dataset, batch_size=8, sampler=cmip_sampler, shuffle=(cmip_sampler is None), num_workers=6, pin_memory=True)

    reanal_dataset = ClimateForecastDataset(DATASET_ROOT, variables, target_var, input_seq_len=12, output_seq_len=12, mode='obs')
    total_len = len(reanal_dataset)
    train_indices = list(range(0, total_len - 156))
    valid_indices = list(range(total_len - 156, total_len - 120))
    test_indices  = list(range(total_len - 120, total_len))

    reanal_train_set = Subset(reanal_dataset, train_indices)
    reanal_valid_set = Subset(reanal_dataset, valid_indices)
    reanal_test_set  = Subset(reanal_dataset, test_indices)

    def _mk_loader(ds, bsz):
        sampler = DistributedSampler(ds) if use_ddp else None
        return DataLoader(ds, batch_size=bsz, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)

    reanal_train_loader = _mk_loader(reanal_train_set, 1)
    reanal_valid_loader = _mk_loader(reanal_valid_set, 1)
    reanal_test_loader  = _mk_loader(reanal_test_set,  1)

    # ====== Model ======
    in_channels = 4
    latent_channels = 32
    encoder = ConvEncoder3D(in_ch=in_channels, fine_ch=latent_channels).to(device)
    decoder = ConvDecoder3D(fine_ch=latent_channels, out_ch=in_channels).to(device)

    wae = AutoencoderWAE(
        encoder=encoder, decoder=decoder,
        lambda_mmd=10.0,         # tune 1~50
        add_enc_noise=False,     # optional
        lr=1e-3,
        use_global_mmd=True      # gather across GPUs for stable MMD
    ).to(device)

    if use_ddp:
        # DDP wrapper is optional since we use custom loops; if wrapping, do it on wae only
        pass

    writer = SummaryWriter(log_dir="./runs/wae") if _is_main() else None
        # ====== Load pretrained weights if available ======
    os.makedirs('./model', exist_ok=True)
    RESUME_PATH = "/data/wuhaotian/diffusionDemo/model/wae_pretrain_best.pt"
    RESUME_PATH = ''
    PRETRAIN_EPOCHS =40  # 默认预训练 20 轮；若成功加载，则跳过预训练

    if os.path.isfile(RESUME_PATH):
        print(f"  Loading pretrained WAE weights from: {RESUME_PATH}")
        # PyTorch 2.6+ 默认 torch.load(weights_only=True)；为兼容老 checkpoint，我们显式允许完整反序列化
        ckpt = torch.load(RESUME_PATH, map_location=device)
        # 兼容 lightning ckpt: 可能包含 'state_dict'
        sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

        # 去除 DataParallel/DDP 的 'module.' 前缀
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[7:]
            new_sd[nk] = v

        missing, unexpected = wae.load_state_dict(new_sd, strict=True)
        # 如果你在结构上有小改动，也可改 strict=False 并查看提示
        if len(missing) or len(unexpected):
            print("  load_state_dict warnings:")
            if len(missing):
                print("  missing keys:", missing)
            if len(unexpected):
                print("  unexpected keys:", unexpected)

        # 成功加载就跳过预训练阶段，直接进入对齐/微调
        PRETRAIN_EPOCHS = 0
        print("✅  Loaded. Will skip Stage 1 pretraining and go to alignment/finetune.")
    else:
        print(f"ℹ️  Pretrain weights not found at {RESUME_PATH}. Will run Stage 1 pretraining.")

    # ====== Stage 1: Pretrain on CMIP ======
    opt_pre = optim.AdamW(wae.parameters(), lr=1e-3, weight_decay=1e-3)
    train_autoencoder_loop(
        cmip_loader,
        reanal_valid_loader,  # or a CMIP valid loader if you have one
        wae,
        opt_pre,
        device,
        writer,
        epochs=20,
        desc="Pretrain_CMIP",
        save_path='./model/wae_pretrain_best.pt',
        local_rank=local_rank
    )

    # ====== Stage 1.5: Optional domain alignment (CMIP ? Reanalysis) ======
    opt_align = optim.AdamW(wae.parameters(), lr=3e-4, weight_decay=1e-3)
    for ep in range(10):
        if use_ddp and isinstance(cmip_loader.sampler, DistributedSampler):
            cmip_loader.sampler.set_epoch(ep)
        if use_ddp and isinstance(reanal_train_loader.sampler, DistributedSampler):
            reanal_train_loader.sampler.set_epoch(ep)
        align_two_domains_epoch(
            wae, opt_align, device,
            cmip_loader, reanal_train_loader,
            lambda_cross=2.0,
            desc=f"Align_CMIP_Reanal [epoch {ep+1}/10]"
        )
        if _is_main():
            torch.save(wae.state_dict(), './model/wae_align_ckpt.pt')

    # ====== Stage 2: Finetune on Reanalysis ======
    opt_ft = optim.AdamW(wae.parameters(), lr=3e-4, weight_decay=1e-3)
    train_autoencoder_loop(
        reanal_train_loader,
        reanal_valid_loader,
        wae,
        opt_ft,
        device,
        writer,
        epochs=100,
        desc="Finetune_Reanal",
        save_path='./model/wae_finetune_best.pt',
        local_rank=local_rank
    )

    if _is_main():
        torch.save(wae.state_dict(), './model/wae_final.pt')
        if writer:
            writer.close()
    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
