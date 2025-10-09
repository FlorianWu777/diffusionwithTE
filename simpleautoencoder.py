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
# -*- coding: utf-8 -*-
"""Autoencoder (3‑D Climate) — channel‑safe rewrite 2025‑06‑21

✓ Encoder / Decoder now keep **fine** and **coarse** latents at configurable
  channel counts and guarantee that every concat / conv layer matches exactly.
✓ Decoder automatically adapts to *fine_ch* ≠ *coarse_ch* (e.g. 16 + 32), so
  you can load older checkpoints without shape errors.
✓ GroupNorm uses the largest divisor ≤ 8 so widening the network never crashes.

You can drop these two classes into your existing training script — nothing
else in `AutoencoderKL` or the training loop has to change.
"""

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def kl_from_standard_normal(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    log_var = torch.clamp(log_var, min=-20.0, max=20.0)
    return 0.5 * (log_var.exp() + mean.pow(2) - 1.0 - log_var).mean()

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

        attn_out, _ = self.attn(x, x, x)      # 不要再写 h, _
        x = self.proj(attn_out) + x

        x = x.view(b, t, h//p, w//p, p, p, c).permute(0, 6, 1, 2, 4, 3, 5)
        return x.contiguous().view(b, c, t, h, w)

# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

class ConvEncoder3D(nn.Module):
    def __init__(self, in_ch: int = 4,
                 fine_ch: int = 32,
                 coarse_ch: int | None = None,
                 patch: int = 2):
        """coarse_ch defaults to fine_ch if not given"""
        super().__init__()
        coarse_ch = coarse_ch or fine_ch

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, stride=(1,2,2), padding=1), nn.GELU(), ResidualBlock3D(8),
            nn.Conv3d(8, 16, 3, stride=(1,2,2), padding=1),  nn.GELU(), ResidualBlock3D(16),
            nn.Conv3d(16, 32,3, stride=(1,2,2), padding=1),  nn.GELU(), ResidualBlock3D(32),
        )  # 输出分辨率 H/8
        self.mom_fine = nn.Conv3d(32, 2*fine_ch, 1)

        self.attn   = PatchAttention3D(32, patch)
        self.down   = nn.Conv3d(32, coarse_ch, 3, stride=(1,2,2), padding=1)  # H/16
        self.mom_coarse = nn.Conv3d(coarse_ch, 2*coarse_ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.net(x)
        mu_f, log_f = torch.chunk(self.mom_fine(h), 2, 1)
        h = self.down(self.attn(h))
        mu_c, log_c = torch.chunk(self.mom_coarse(h), 2, 1)
        # fine, coarse 顺序固定
        #return mu_f, log_f, mu_c, log_c
        return mu_c, log_c
# -----------------------------------------------------------------------------
# Decoder (adaptable fine/coarse)
# -----------------------------------------------------------------------------

class ConvDecoder3D(nn.Module):
    def __init__(self, fine_ch: int = 32, coarse_ch: int | None = None,
                 out_ch: int = 4, patch: int = 2):
        super().__init__()
        coarse_ch = coarse_ch or fine_ch

        # coarse latent → upsample to match fine spatial scale (H/16 → H/8)
        self.up   = nn.ConvTranspose3d(coarse_ch, coarse_ch, (1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.to_fuse = nn.Conv3d(coarse_ch, coarse_ch, 1)
        self.attn = PatchAttention3D(coarse_ch, patch)

        # Decoder tower mirrors encoder (channels: 64→32→16→8)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(coarse_ch, 64, (1,4,4), stride=(1,2,2), padding=(0,1,1)), nn.GELU(), ResidualBlock3D(64),
            nn.ConvTranspose3d(64, 32, (1,4,4), stride=(1,2,2), padding=(0,1,1)),  nn.GELU(), ResidualBlock3D(32),
            nn.ConvTranspose3d(32, 16, (1,4,4), stride=(1,2,2), padding=(0,1,1)),  nn.GELU(), ResidualBlock3D(16),
            nn.Conv3d(16, out_ch, 3, padding=1)
        )
        self.act = nn.GELU()

    def forward(self,  z_coarse):
        z = self.up(z_coarse)
        #z = torch.cat([z, z_fine], dim=1)
        z = self.act(self.to_fuse(z))
        z = self.attn(z)
        return self.dec(z)


# -----------------------------------------------------------------------------
# Autoencoder‑KL with SD‑style scaling
# -----------------------------------------------------------------------------

class AutoencoderKL_dual(pl.LightningModule):
    scaling_factor: float = 0.18215  # public attr so diffusion can access

    def __init__(self, encoder: nn.Module, decoder: nn.Module, kl_weight: float = 0.003,
                 kl_warmup_epochs: int = 20):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.base_kl_weight = kl_weight
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.hidden_width = 32
        self.latent_channels =32

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _sample(mean: torch.Tensor, log_var: torch.Tensor, posterior: bool = True) -> torch.Tensor:
        if not posterior:
            return mean
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mean + eps * std

    # ------------------------------------------------------------------ fwd
    def forward(self, x: torch.Tensor, *, posterior: bool = True):
        mu1, logsigma1, mu2, logsigma2 = self.encoder(x)
        z1 = self._sample(mu1, logsigma1, posterior)
        z2 = self._sample(mu2, logsigma2, posterior)

        # scale latents before feeding to diffusion / decoder
        z1_s = z1 * self.scaling_factor
        z2_s = z2 * self.scaling_factor
        recon = self.decode(z1_s, z2_s)
        return recon, (mu1, logsigma1, mu2, logsigma2)

    def encode(self, x, *_, **__):
        """Return (mu_f, log_f, mu_c, log_c) exactly like self.encoder."""
        return self.encoder(x)
    def decode(self, z1_s: torch.Tensor, z2_s: torch.Tensor) -> torch.Tensor:
        z1 = z1_s / self.scaling_factor
        z2 = z2_s / self.scaling_factor
        return self.decoder(z1, z2)

    # ------------------------------------------------------------------ loss
    def _loss(self, batch):
        x, = batch  # dataset yields (x,) or (x,y) where y==x
        recon, stats = self(x)
        mu1, logsigma1, mu2, logsigma2 = stats
        rec_loss = torch.mean(torch.abs(x - recon))
        kl = kl_from_standard_normal(mu1, logsigma1) + kl_from_standard_normal(mu2, logsigma2)
        total = rec_loss + self.kl_weight * kl
        return total, rec_loss, kl

    def update_kl_weight(self, epoch: int):
        """Call once per epoch when you train manually."""
        if epoch < self.kl_warmup_epochs:
            self.kl_weight = (epoch + 1) / self.kl_warmup_epochs * self.base_kl_weight
        else:
            self.kl_weight = self.base_kl_weight
    # ------------------------------------------------------------------ steps
    def training_step(self, batch, _):
        if self.current_epoch < self.kl_warmup_epochs:
            self.kl_weight = (self.current_epoch + 1) / self.kl_warmup_epochs * self.base_kl_weight
        else:
            self.kl_weight = self.base_kl_weight
        tot, rec, kl = self._loss(batch)
        self.log_dict({"train_loss": tot, "train_rec": rec, "train_kl": kl})
        return tot

    def validation_step(self, batch, _):
        tot, rec, kl = self._loss(batch)
        self.log_dict({"val_loss": tot, "val_rec": rec, "val_kl": kl}, prog_bar=True)

    # ------------------------------------------------------------------ opt
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.25, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_rec"}}

class AutoencoderKL(pl.LightningModule):
    scaling_factor: float = 0.18215

    def __init__(self, encoder: nn.Module, decoder: nn.Module, kl_weight: float = 0.003,
                 kl_warmup_epochs: int = 20):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.base_kl_weight = kl_weight
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.hidden_width = 32
        self.latent_channels =32
    @staticmethod
    def _sample(mean: torch.Tensor, log_var: torch.Tensor, posterior: bool = True) -> torch.Tensor:
        if not posterior:
            return mean
        std = (0.5 * log_var).exp()
        return mean + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor, *, posterior: bool = True):
        #mu_f, log_f, mu_c, log_c = self.encoder(x)
        mu_c, log_c = self.encoder(x)
        # 忽略 fine latent
        z_c = self._sample(mu_c, log_c, posterior)
        z_c_scaled = z_c * self.scaling_factor

        # 提供零张量占位给 fine latent，保证 decoder 不报错
        #z_f_dummy = torch.zeros_like(z_c_scaled)

        #recon = self.decode(z_f_dummy, z_c_scaled)
        recon = self.decode(z_c_scaled)
        return recon, (mu_c, log_c)

    def encode(self, x, *_, **__):
        mu_c, log_c = self.encoder(x)
        return mu_c, log_c

    def decode(self, z_coarse: torch.Tensor) -> torch.Tensor:
        return self.decoder( z_coarse / self.scaling_factor)

    def _loss(self, batch):
        x, = batch
        recon, stats = self(x)
        mu, log_var = stats
        rec_loss = torch.mean(torch.abs(x - recon))
        kl = kl_from_standard_normal(mu, log_var)
        total = rec_loss + self.kl_weight * kl
        return total, rec_loss, kl

    def update_kl_weight(self, epoch: int):
        if epoch < self.kl_warmup_epochs:
            self.kl_weight = (epoch + 1) / self.kl_warmup_epochs * self.base_kl_weight
        else:
            self.kl_weight = self.base_kl_weight

    def training_step(self, batch, _):
        self.update_kl_weight(self.current_epoch)
        tot, rec, kl = self._loss(batch)
        self.log_dict({"train_loss": tot, "train_rec": rec, "train_kl": kl})
        return tot

    def validation_step(self, batch, _):
        tot, rec, kl = self._loss(batch)
        self.log_dict({"val_loss": tot, "val_rec": rec, "val_kl": kl}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.25, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_rec"}}
# -----------------------------------------------------------------------------
# training / evaluation helpers (DDP‑friendly standalone loops)
# -----------------------------------------------------------------------------

def train_autoencoder_loop(
    train_loader,                # 训练 DataLoader
    val_loader,                  # 验证 DataLoader
    model,                       # AutoencoderKL or DDP-wrapped
    optimizer,
    device,
    writer,                      # TensorBoard SummaryWriter 或 None
    epochs: int,
    desc: str,
    save_path: str,
    local_rank: int = 0          # 分布式时方便判定主进程
):
    best_val = float("inf")

    for epoch in range(epochs):
        # 1) DDP 的 epoch shuffle
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        # 告诉模型当前 epoch -> 用于 KL-warm-up
        #   只有原始模型需要，DDP 封装时也要 model.module
        core = model.module if hasattr(model, "module") else model
        core.update_kl_weight(epoch)      # ← 代替 core.current_epoch = ep


        prog = tqdm(train_loader,
                    desc=f"[{desc}] Epoch {epoch+1}/{epochs}",
                    disable=(local_rank != 0))

        running = {"tot": 0.0, "rec": 0.0, "kl": 0.0}

        for x, *_ in prog:
            x = x.to(device)

            # 2) 统一计算损失
            tot, rec, kl = model._loss((x,))

            optimizer.zero_grad()
            tot.backward()
            optimizer.step()

            # 3) 统计
            running["tot"] += tot.item()
            running["rec"] += rec.item()
            running["kl"]  += kl.item()

        # 4) Logging（仅主进程）
        if writer and local_rank == 0:
            n = len(train_loader)
            writer.add_scalars(f"{desc}/train",
                               {k: v / n for k, v in running.items()},
                               epoch)
            writer.add_scalar(f"{desc}/kl_weight",
                              (model.module if hasattr(model, 'module') else model).kl_weight,
                              epoch)

        # 5) 验证（主进程）
        if local_rank == 0:
            val_tot, val_rec, val_kl = evaluate_autoencoder(val_loader, model, device)

            if writer:
                writer.add_scalars(f"{desc}/val",
                                   {"tot": val_tot, "rec": val_rec, "kl": val_kl},
                                   epoch)

            # 6) 保存最优
            if val_tot < best_val:
                best_val = val_tot
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state, save_path)
                print(f"✅  Best model saved @ epoch {epoch+1}  val_tot={val_tot:.4f}")

def evaluate_autoencoder(loader, model: nn.Module, device: torch.device) -> float:
    model.eval()
    total = rec_total = kl_total = 0.0
    with torch.no_grad():
        for batch in loader:
            x, = (b.to(device) for b in batch[:1])
            recon, (mu2, logsigma2) = model(x)
            rec = torch.mean(torch.abs(x - recon)).item()
            kl =  kl_from_standard_normal(mu2, logsigma2).item()
            total += rec + 0.003 * kl
            rec_total += rec
            kl_total += kl
    n = len(loader)
    return total / n, rec_total / n, kl_total / n
    
def main():
# 设置模型参数
    in_channels = 4          # 输入变量数量
    latent_channels = 32      # 降维后通道数
    kl_weight = 0.003
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    
    encoder = ConvEncoder3D(in_ch=in_channels, fine_ch=latent_channels).to(device)
    decoder = ConvDecoder3D(fine_ch=latent_channels, out_ch=in_channels).to(device)
    
    autoencoder_model = AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        kl_weight=kl_weight,        # ← 你原来就有
        kl_warmup_epochs=20         # ← 默认 20，也可以省略
    ).to(device)
#    autoencoder_model = DDP(autoencoder_model, device_ids=[local_rank], find_unused_parameters=True)
    
    

    writer = SummaryWriter(log_dir="./runs/autoencoder") if local_rank == 0 else None
    
    optimizer = optim.AdamW(autoencoder_model.parameters(), lr=1e-3, weight_decay=1e-3)
    
    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    
    cmip_names = ['EC-Earth3/r2i1p1f1',
        'EC-Earth3/r7i1p1f1',
        'EC-Earth3/r10i1p1f1',
        'EC-Earth3/r12i1p1f1',
        'EC-Earth3/r14i1p1f1',
        'MRI-ESM2-0/r1i1p1f1',
        'MRI-ESM2-0/r2i1p1f1',
        'MRI-ESM2-0/r3i1p1f1',
        'MRI-ESM2-0/r4i1p1f1',
        'MRI-ESM2-0/r5i1p1f1']
    cmip_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names)
    cmip_sampler = DistributedSampler(cmip_dataset, shuffle=True, drop_last=True)
    cmip_loader = DataLoader(cmip_dataset, batch_size=16, sampler=cmip_sampler, num_workers=6)
    
    reanal_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'obs')
    total_len = len(reanal_dataset)
    
    from torch.utils.data import Subset
    train_indices = list(range(0, total_len - 156))
    valid_indices = list(range(total_len - 156, total_len - 120))
    test_indices = list(range(total_len - 120, total_len))
    
    reanal_train_set = Subset(reanal_dataset, train_indices)
    reanal_valid_set = Subset(reanal_dataset, valid_indices)
    reanal_test_set = Subset(reanal_dataset, test_indices)
    
    reanal_train_sampler = DistributedSampler(reanal_train_set)
    reanal_valid_sampler = DistributedSampler(reanal_valid_set)
    reanal_test_sampler = DistributedSampler(reanal_test_set)
    
    reanal_train_loader = DataLoader(reanal_train_set, batch_size=2, sampler=reanal_train_sampler, num_workers=0)
    reanal_valid_loader = DataLoader(reanal_valid_set, batch_size=2, sampler=reanal_valid_sampler, num_workers=0)
    reanal_test_loader = DataLoader(reanal_test_set, batch_size=2, sampler=reanal_test_sampler, num_workers=0)
   # ckpt_path = "./model/simpleautoencoder_ptbest.pt"
   # autoencoder_model.load_state_dict(torch.load(ckpt_path), strict=True)  # 不要 strict=False！  
    # Train the Autoencoder
   # train_autoencoder_loop(
    #    cmip_loader,
     #   reanal_valid_loader,
      #  autoencoder_model,
      #  optimizer,
       # device,
       # writer,
       # epochs=80,
       # desc="Autoencoder Training",
       # save_path='./model/simpleautoencoder_ptbest.pt'
   # )
    
        # 1. 不包 DDP，先构造原始模型
    autoencoder_pt = AutoencoderKL(encoder, decoder, kl_weight=kl_weight).to(device)
    opt_pre = optim.AdamW(autoencoder_pt.parameters(), lr=1e-3, weight_decay=1e-3)
    core = AutoencoderKL(encoder, decoder, kl_weight=kl_weight).to(device)
    core.load_state_dict(torch.load("./model/ae_pretrain_best.pt", map_location=device))
    train_autoencoder_loop(
        cmip_loader,                  # 预训练数据
        reanal_valid_loader,          # 也可以用 cmip_valid_loader
        autoencoder_pt,
        opt_pre,
        device,
        writer if local_rank==0 else None,
        epochs=20,
        desc="Pretrain",
        save_path="./model/ae_pretrain_best.pt",
        local_rank=local_rank
    )
    
    # ② 微调阶段（Reanalysis）
    core = AutoencoderKL(encoder, decoder, kl_weight=kl_weight).to(device)
    core.load_state_dict(torch.load("./model/ae_pretrain_best.pt", map_location=device))
    #autoencoder_ft = DDP(core, device_ids=[local_rank], find_unused_parameters=True)
    autoencoder_ft = core
    opt_ft = optim.AdamW(autoencoder_ft.parameters(), lr=3e-4, weight_decay=1e-3)
    
    train_autoencoder_loop(
        reanal_train_loader,          # 微调数据
        reanal_valid_loader,
        autoencoder_ft,
        opt_ft,
        device,
        writer if local_rank==0 else None,
        epochs=100,
        desc="Finetune",
        save_path="./model/vae_finetune_best.pt",
        local_rank=local_rank
    )
    if local_rank == 0:
        torch.save(autoencoder_model.state_dict(), 'autoencoder_final.pt')
    
    if writer:
        writer.close()
    dist.destroy_process_group()
        
if __name__ == "__main__":
    main()   
    
    
    
