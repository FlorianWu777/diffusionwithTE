from __future__ import annotations # 确保类型提示兼容
import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from dataloader import ClimateForecastDataset
from trainvqvae import ClimateForecastDataset # 假设您的 dataloader 在这里
from annual_embedding import YearAdditiveEmbedding
from typing import Optional, Iterable, Tuple # 确保导入
from vaeautoencoder import AutoencoderKL
from vaeblock import SimpleConvEncoder, SimpleConvDecoder
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- VAE 原始辅助函数 ---

def kl_from_standard_normal(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()

# --- VAE 原始训练循环 ---

def train_autoencoder_loop(dataloader, valid_loader, autoencoder_model, optimizer, device, writer, epochs, desc, save_path):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)
        autoencoder_model.train()

        for batch in tqdm(dataloader, desc=f"[{desc}] Epoch {epoch+1}", disable=dist.get_rank() != 0):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, _, meta = batch
            else:
                x, _ = batch
                meta = None
            x = x.to(device)

            month_idx = None
            if isinstance(meta, dict) and ('month_in' in meta):
                month_idx = meta['month_in'].to(device)

            (x_recon, mean, log_var) = autoencoder_model(x, month_idx=month_idx)

            rec_loss = (x - x_recon).abs().mean()
            kl_loss = kl_from_standard_normal(mean, log_var)
            kl_w = autoencoder_model.module.ae.kl_weight if hasattr(autoencoder_model.module, "ae") else autoencoder_model.module.kl_weight
            loss = rec_loss + kl_w * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if dist.get_rank() == 0:
            if writer is not None:
                writer.add_scalar(f"{desc}/Recon_Loss", rec_loss.item(), epoch)
                writer.add_scalar(f"{desc}/KL_Loss", kl_loss.item(), epoch)
            print(f"[{desc}] Epoch {epoch+1}: Recon={rec_loss.item():.4f}, KL={kl_loss.item():.4f}")

            val_loss = evaluate_autoencoder(valid_loader, autoencoder_model, device)
            if writer is not None:
                writer.add_scalar(f"{desc}/Val_Loss", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(autoencoder_model.module.state_dict(), save_path)
                print(f"✅ Best Autoencoder saved at epoch {epoch+1} with Val_Loss={val_loss:.4f}")

# --- VAE 原始评估函数 ---

def evaluate_autoencoder(valid_loader, autoencoder_model, device):
    autoencoder_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, _, meta = batch
            else:
                x, _ = batch
                meta = None
            x = x.to(device)
            month_idx = None
            if isinstance(meta, dict) and ('month_in' in meta):
                month_idx = meta['month_in'].to(device)
            
            # 评估时只关心重建损失
            (x_recon, _, _) = autoencoder_model(x, month_idx=month_idx)
            rec_loss = (x - x_recon).abs().mean()
            total_loss += rec_loss.item()

    return total_loss / len(valid_loader)

# ---------------------------------------------------------------------
# ★★★ 新增：对齐所需的辅助函数 (来自 WAE 脚本) ★★★
# ---------------------------------------------------------------------

def _is_main() -> bool:
    """检查是否为主进程"""
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

@torch.no_grad()
def _next(loader_iter, fallback_iter):
    """安全地从迭代器取下一个 batch，失败则从备用迭代器取"""
    try:
        return next(loader_iter)
    except StopIteration:
        return next(fallback_iter)

def all_gather_concat(t: torch.Tensor) -> torch.Tensor:
    """在 DDP 环境中收集所有 GPU 上的张量并拼接"""
    if not dist.is_available() or not dist.is_initialized():
        return t
    world = dist.get_world_size()
    if world == 1:
        return t
    tensors = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(tensors, t.contiguous())
    return torch.cat(tensors, dim=0)

def _flatten_feat(t: torch.Tensor) -> torch.Tensor:
    """将潜变量展平 [B, C, T, H, W] -> [B, -1]"""
    return t.view(t.shape[0], -1)

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: Optional[Iterable[float]] = None) -> torch.Tensor:
    """MMD^2 with multi-bandwidth RBF kernel (biased estimate)."""
    x = _flatten_feat(x)
    y = _flatten_feat(y)

    if sigmas is None:
        sigmas = [0.5, 1., 2., 4., 8.] # 简化的 sigma 列表

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

# ---------------------------------------------------------------------
# ★★★ 新增：适用于 VAE 的对齐循环 ★★★
# ---------------------------------------------------------------------

def align_vae_domains_epoch(model: nn.Module, opt: torch.optim.Optimizer, device,
                            cmip_loader: DataLoader, reanal_loader: DataLoader,
                            lambda_cross: float = 1.0, # MMD 对齐的权重
                            desc: str = "Align VAE"):
    model.train()
    cmip_it = iter(cmip_loader)
    rean_it = iter(reanal_loader)
    cmip_backup = iter(cmip_loader)
    rean_backup = iter(reanal_loader)

    steps = min(len(cmip_loader), len(reanal_loader))
    prog = tqdm(range(steps), desc=desc, disable=not _is_main())

    for _ in prog:
        # 1. 加载 CMIP 数据
        batch_c = _next(cmip_it, cmip_backup)
        if isinstance(batch_c, (list, tuple)) and len(batch_c) == 3:
            x_c, _, meta_c = batch_c
        else:
            x_c, _ = batch_c; meta_c = None
        x_c = x_c.to(device)
        month_c = meta_c['month_in'].to(device) if isinstance(meta_c, dict) and 'month_in' in meta_c else None
        
        # 2. 加载 Reanalysis 数据
        batch_r = _next(rean_it, rean_backup)
        if isinstance(batch_r, (list, tuple)) and len(batch_r) == 3:
            x_r, _, meta_r = batch_r
        else:
            x_r, _ = batch_r; meta_r = None
        x_r = x_r.to(device)
        month_r = meta_r['month_in'].to(device) if isinstance(meta_r, dict) and 'month_in' in meta_r else None

        # 3. 前向传播
        (recon_c, mean_c, log_var_c) = model(x_c, month_idx=month_c)
        (recon_r, mean_r, log_var_r) = model(x_r, month_idx=month_r)

        # 4. 计算 VAE 损失
        kl_w = model.module.ae.kl_weight if hasattr(model.module, "ae") else model.module.kl_weight
        
        rec_c = (x_c - recon_c).abs().mean()
        kl_c = kl_from_standard_normal(mean_c, log_var_c)
        loss_c = rec_c + kl_w * kl_c

        rec_r = (x_r - recon_r).abs().mean()
        kl_r = kl_from_standard_normal(mean_r, log_var_r)
        loss_r = rec_r + kl_w * kl_r
        
        # 5. 计算跨域对齐损失 (在 mean 上做 MMD)
        # DDP 同步：收集所有 GPU 上的 mean
        mean_c_eval = all_gather_concat(mean_c)
        mean_r_eval = all_gather_concat(mean_r)
        
        # 注意：这里我们只在 VAE 的 mean 上做 MMD 对齐
        mmd_cross = mmd_rbf(mean_c_eval, mean_r_eval)
        
        # 6. 总损失
        # VAE 损失 + 跨域对齐损失
        total_loss = (loss_c + loss_r) * 0.5 + lambda_cross * mmd_cross

        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        opt.step()

        if _is_main():
            prog.set_postfix(
                rec_c=float(rec_c.detach()),
                rec_r=float(rec_r.detach()),
                kl_c=float(kl_c.detach()),
                kl_r=float(kl_r.detach()),
                mmd=float(mmd_cross.detach())
            )

# --- VAE 原始包装模块 ---
from typing import Optional # 确保导入 Optional

class AEWithTE(nn.Module):
    """
    一个包装器，将自编码器（ae）与年循环时间嵌入（te）结合起来。
    """
    def __init__(self, ae: nn.Module, te: nn.Module, channel_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.ae = ae
        self.te = te
        self.register_buffer("channel_mask", channel_mask if channel_mask is not None else None)

    def forward(self, x, month_idx=None):
        """前向传播，先应用时间嵌入，再通过自编码器。"""
        x_te = self.te(x, month_idx=month_idx)
        if self.channel_mask is not None:
            # 仅在 mask=1 的通道上生效
            x_in = x + (x_te - x) * self.channel_mask
        else:
            x_in = x_te
        return self.ae(x_in)

    # ★★★ 新增的核心代码 ★★★
    def __getattr__(self, name: str):
        """
        属性查找的后备方法。
        当在 AEWithTE 实例上找不到像 'hidden_width' 这样的属性时，
        此方法会自动尝试从内部的 self.ae 对象获取。
        """
        try:
            # 首先尝试在父类（nn.Module）中查找，以避免覆盖其内部方法
            return super().__getattr__(name)
        except AttributeError:
            # 如果父类中没有，就从 self.ae 查找
            if hasattr(self.ae, name):
                return getattr(self.ae, name)
            # 如果 self.ae 也没有，则抛出原始的 AttributeError
            raise
#
## --- VAE 原始 Main 函数 (已修改训练流程) ---
#
#local_rank = int(os.environ.get("LOCAL_RANK", 0)) # 兼容单卡
#use_ddp = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
#
#if use_ddp:
#    torch.cuda.set_device(local_rank)
#    dist.init_process_group(backend="nccl")
#device = torch.device(f"cuda:{local_rank}")
#
#writer = SummaryWriter(log_dir="./runs/autoencoder") if _is_main() else None
#
#encoder = SimpleConvEncoder(in_dim=4, levels=2, min_ch=64).to(device)
#decoder = SimpleConvDecoder(in_dim=4, levels=2, min_ch=64).to(device)
#ae_core = AutoencoderKL(
#    encoder=encoder,
#    decoder=decoder,
#    kl_weight=0.01,
#    encoded_channels=64,
#    hidden_width=32,
#).to(device)
#
#channel_mask = None
#te = YearAdditiveEmbedding(num_channels=4, init_zero=True).to(device)
#autoencoder_model = AEWithTE(ae_core, te, channel_mask=channel_mask).to(device)
#
#if use_ddp:
#    autoencoder_model = DDP(autoencoder_model, device_ids=[local_rank], find_unused_parameters=False) # ★find_unused_parameters 必须为 False
#
## --- 数据加载 ---
#root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
#variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
#target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
#
#cmip_names = ['EC-Earth3/r2i1p1f1','EC-Earth3/r7i1p1f1','EC-Earth3/r10i1p1f1','EC-Earth3/r12i1p1f1','EC-Earth3/r14i1p1f1',
#              'MRI-ESM2-0/r1i1p1f1', 'MRI-ESM2-0/r2i1p1f1','MRI-ESM2-0/r3i1p1f1','MRI-ESM2-0/r4i1p1f1','MRI-ESM2-0/r5i1p1f1']
## ★★★ 确保开启 return_meta=True ★★★
#cmip_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names, return_meta=True)
#cmip_sampler = DistributedSampler(cmip_dataset, shuffle=True, drop_last=True) if use_ddp else None
#cmip_loader = DataLoader(cmip_dataset, batch_size=4, sampler=cmip_sampler, num_workers=4, shuffle=(cmip_sampler is None))
#
#reanal_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'obs', return_meta=True)
#total_len = len(reanal_dataset)
#
#train_indices = list(range(0, total_len - 156))
#valid_indices = list(range(total_len - 156, total_len - 120))
#test_indices = list(range(total_len - 120, total_len))
#
#reanal_train_set = Subset(reanal_dataset, train_indices)
#reanal_valid_set = Subset(reanal_dataset, valid_indices)
#reanal_test_set = Subset(reanal_dataset, test_indices)
#
#reanal_train_sampler = DistributedSampler(reanal_train_set) if use_ddp else None
#reanal_valid_sampler = DistributedSampler(reanal_valid_set) if use_ddp else None
#
#reanal_train_loader = DataLoader(reanal_train_set, batch_size=2, sampler=reanal_train_sampler, num_workers=0, shuffle=(reanal_train_sampler is None))
#reanal_valid_loader = DataLoader(reanal_valid_set, batch_size=2, sampler=reanal_valid_sampler, num_workers=0, shuffle=(reanal_valid_sampler is None))
#
## ---------------------------------------------------------------------
## ★★★ 修改后的训练流程 ★★★
## ---------------------------------------------------------------------
#
## ====== 阶段 1: 预训练 Autoencoder (在 CMIP 上) ======
#print("--- STAGE 1: Pretraining on CMIP ---")
#optimizer_pre = optim.AdamW(autoencoder_model.parameters(), lr=1e-4, weight_decay=1e-3)
#train_autoencoder_loop(
#    cmip_loader,
#    reanal_valid_loader,
#    autoencoder_model,
#    optimizer_pre,
#    device,
#    writer,
#    epochs=50, # 预训练轮数
#    desc="VAE Pretraining (CMIP)",
#    save_path='./model/vae_pretrain_best.pt'
#)
#
## ====== 阶段 1.5: 对齐 (CMIP <-> Reanalysis) ======
#print("\n--- STAGE 1.5: Aligning CMIP and Reanalysis latent spaces ---")
#optimizer_align = optim.AdamW(autoencoder_model.parameters(), lr=5e-5, weight_decay=1e-3) # 使用稍小的学习率
#ALIGN_EPOCHS = 20 # 对齐轮数
#for epoch in range(ALIGN_EPOCHS):
#    # 确保 DDP 采样器设置 epoch
#    if use_ddp:
#        cmip_loader.sampler.set_epoch(epoch)
#        reanal_train_loader.sampler.set_epoch(epoch)
#        
#    align_vae_domains_epoch(
#        autoencoder_model,
#        optimizer_align,
#        device,
#        cmip_loader,
#        reanal_train_loader,
#        lambda_cross=1.0, # 跨域 MMD 损失的权重
#        desc=f"[Align VAE] Epoch {epoch+1}/{ALIGN_EPOCHS}"
#    )
#    # (对齐阶段也可以加入验证，这里简化)
#    if _is_main():
#        torch.save(autoencoder_model.module.state_dict(), './model/vae_aligned.pt')
#
#
## ====== 阶段 2: 微调 (在 Reanalysis 上) ======
#print("\n--- STAGE 2: Finetuning on Reanalysis ---")
#optimizer_ft = optim.AdamW(autoencoder_model.parameters(), lr=1e-5, weight_decay=1e-3) # 使用更小的学习率
#train_autoencoder_loop(
#    reanal_train_loader,
#    reanal_valid_loader,
#    autoencoder_model,
#    optimizer_ft, # ★ 注意使用 finetune_opt
#    device,
#    writer,
#    epochs=50, # 微调轮数
#    desc="VAE Finetune (Reanal)",
#    save_path='./model/vae_finetune_best.pt'
#)
#
## --- 清理 ---
#if _is_main():
#    print("--- Training complete. Saving final models. ---")
#    # 建议分别保存 AE 与 TE，便于下游组合复用
#    torch.save(autoencoder_model.module.ae.state_dict(), 'autoencoder_final_ae.pt')
#    torch.save(autoencoder_model.module.te.state_dict(), 'autoencoder_final_te.pt')
#
#if writer:
#    writer.close()
#if use_ddp:
#    dist.destroy_process_group()