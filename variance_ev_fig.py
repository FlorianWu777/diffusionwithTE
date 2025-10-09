#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_and_spread_skill.py
- 合并：单编码器 AFNO + 3D-UNet 海冰扩散模型 推理 + 直接计算并输出
  error variance (MSE) 与 ensemble spread 随 lead(t) 的曲线。
- 支持两种入口：
  1) 正常推理并在线累计统计（更省内存，不落大体积 .pt）
  2) 从已保存的 (N, E, 1, T, H, W) .pt 文件直接计算（第 0 个为 GT，后面为成员）

误差定义：error variance = MSE( ensemble mean , truth )
spread 定义：ensemble variance（跨成员方差）
均按空间平均、样本平均 -> 得到每个 lead 的标量。
"""

import argparse
import gc
import os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.distributed as dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

# ======== 3rd-party / project imports (与原训练一致) ========
from simpleautoencoder import ConvEncoder3D, ConvDecoder3D, AutoencoderKL
from models.genforecast import analysis, unet, training
from trainvqvae import ClimateForecastDataset
from models.diffusion.diffusion import LatentDiffusion


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run inference, and directly output error/spread curves vs lead time."
    )
    # 推理相关
    p.add_argument("--checkpoint", type=str, default="/data/wuhaotian/diffusionDemo/model/epoch=epoch=65-val_loss=val_loss=0.1531.ckpt",
                   help="Lightning checkpoint (.ckpt). 若仅从已有 .pt 评估可不填。")
    p.add_argument("--data_root", type=str, default="/data/wuhaotian/diffusionDemo/dataset1/",
                   help="数据根目录（与训练一致）。")
    p.add_argument("--future_steps", type=int, default=12,
                   help="模型预测的 lead 月数。")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda, cuda:1, 或 cpu。")
    p.add_argument("--ensemble_size", type=int, default=20,
                   help="每个样本的集合成员数（不含 GT）。")
    p.add_argument("--var_index", type=int, default=1,
                   help="评估的变量通道索引，默认 1 (siconca/abs)。")
    p.add_argument("--lead_slice", type=str, default="0:12",
                   help="对时间维的 Python 切片字符串（如 '1::12' 与你现有代码一致）。")

    # 输出与复用
    p.add_argument("--out_dir", default="/share/wuhaotian/inference_outputs",
                   help="输出目录：图与 CSV 存在此处，若推理也会把中间件存在此处。")
    p.add_argument("--from_saved_pt", type=str, default="",
                   help="若提供，则直接从 .pt (N, E, 1, T, H, W; E 含 GT at 0) 计算曲线。")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"],
                   help="推理/统计使用的分割。仅在未提供 --from_saved_pt 时生效。")
    return p.parse_args()


# ---------------------- Data ----------------------
class SeaIceDataModule(pl.LightningDataModule):
    """Creates train/val/test loaders identical to training setup."""

    def __init__(self, root_dir: str, batch_size: int = 32, future_steps: int = 12):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.future_steps = future_steps

    def setup(self, stage: Optional[str] = None):
        variables = target_var = [
            "psl/anom", "siconca/abs", "tas/anom", "tos/anom"
        ]
        cmip_names = [
            "EC-Earth3/r2i1p1f1", "EC-Earth3/r7i1p1f1", "EC-Earth3/r10i1p1f1",
            "EC-Earth3/r12i1p1f1", "EC-Earth3/r14i1p1f1", "MRI-ESM2-0/r1i1p1f1",
            "MRI-ESM2-0/r2i1p1f1", "MRI-ESM2-0/r3i1p1f1", "MRI-ESM2-0/r4i1p1f1",
            "MRI-ESM2-0/r5i1p1f1",
        ]
        self.cmip_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "transfer", cmip_names
        )
        self.reanal_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "obs"
        )
        total_len = len(self.reanal_dataset)
        train_cutoff, valid_cutoff = total_len - 360, total_len - 348
        train_idx = list(range(0, train_cutoff))
        valid_idx = list(range(train_cutoff, valid_cutoff))
        test_idx = list(range(valid_cutoff, total_len))
        self.reanal_train_set = Subset(self.reanal_dataset, train_idx)
        self.reanal_valid_set = Subset(self.reanal_dataset, valid_idx)
        self.reanal_test_set = Subset(self.reanal_dataset, test_idx)

    def _dl(self, dataset, shuffle: bool, batch_size: Optional[int] = None):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size,
            sampler=sampler, num_workers=4, pin_memory=True, shuffle=False
        )

    def val_dataloader(self):
        return self._dl(self.reanal_valid_set, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.reanal_test_set, shuffle=False)


# ---------- Network recreation ----------
def setup_singleenc_model(
    past_timesteps: int = 12,
    future_timesteps: int = 12,
    autoenc_ckpt: str = "/data/wuhaotian/diffusionDemo/model/vae_finetune_best.pt",
    model_dir: str = "./model/genforecast_single",
    lr: float = 1e-4,
    local_rank: int = 0,
):
    world_size = torch.cuda.device_count() or 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    in_channels = 4
    latent_channels = 32
    kl_weight = 0.003

    encoder = ConvEncoder3D(in_ch=in_channels, fine_ch=latent_channels).to(device)
    decoder = ConvDecoder3D(fine_ch=latent_channels, out_ch=in_channels).to(device)

    autoenc = AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        kl_weight=kl_weight,
        kl_warmup_epochs=20
    ).to(device)
    autoenc.load_state_dict(torch.load(autoenc_ckpt, map_location="cpu"), strict=False)
    autoenc.to(device).eval()
    for p in autoenc.parameters():
        p.requires_grad_(False)

    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoder=autoenc,
        embed_dim=64,
        analysis_depth=4,
        cascade_depth=3,
        input_patches=(1,),
        input_size_ratios=(1,),
        output_patches=future_timesteps,
        train_autoenc=False,
    )

    unet_model = unet.UNetModel(
        in_channels=autoenc.hidden_width,
        out_channels=autoenc.hidden_width,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(1, 2),
        dims=3,
        channel_mult=(1, 2, 4),
        num_heads=4,
        num_timesteps=future_timesteps,
        context_ch=analysis_net.cascade_dims,
        use_checkpoint=True,
    )

    ldm, trainer = training.setup_genforecast_training(
        unet_model,
        autoenc,
        context_encoder=analysis_net,
        model_dir=model_dir,
        lr=lr,
        num_gpus=world_size,
        inference_mode=True,
    )
    gc.collect()
    return ldm, trainer


# ---------- Utils ----------
def parse_slice(slice_str: str, max_len: Optional[int] = None) -> slice:
    """
    将 '1::12' 这类字符串转为 slice 对象。格式：start:stop:step；任一可省略。
    """
    parts = slice_str.split(":")
    parts += [""] * (3 - len(parts))
    start = int(parts[0]) if parts[0] != "" else None
    stop = int(parts[1]) if parts[1] != "" else None
    step = int(parts[2]) if parts[2] != "" else None
    return slice(start, stop, step)


def update_running_stats(
    mse_sum: torch.Tensor,
    spread_sum: torch.Tensor,
    y_t_hw: torch.Tensor,            # (T, H, W)
    preds_members_t_hw: torch.Tensor # (E, T, H, W)
):
    """
    对单个样本更新累计量：
    - error variance = MSE(ensemble mean, truth)
    - spread         = VAR_across_members
    均按空间平均 -> 得到每个 lead 的标量后再累加。
    """
    ens_mean = preds_members_t_hw.mean(dim=0)         # (T, H, W)
    ens_var  = preds_members_t_hw.var(dim=0, unbiased=False)  # (T, H, W)

    sqerr = (ens_mean - y_t_hw)**2                    # (T, H, W)
    mse_per_lead = sqerr.mean(dim=(1, 2))             # (T,)
    spread_per_lead = ens_var.mean(dim=(1, 2))        # (T,)

    mse_sum += mse_per_lead.cpu()
    spread_sum += spread_per_lead.cpu()


def compute_from_saved_pt(pt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从已保存的预测张量 (N, E, 1, T, H, W) 直接计算曲线。
    E 维中第 0 个应为 GT，1..E-1 为成员。
    """
    tensor = torch.load(pt_path, map_location="cpu")
    assert tensor.ndim == 6, f"Expect (N,E,1,T,H,W), got {tensor.shape}"
    N, E, C1, T, H, W = tensor.shape
    assert C1 == 1, "通道维应为 1（已在保存时抽取了 var_index）。"

    mse_sum = torch.zeros(T)
    spread_sum = torch.zeros(T)

    y_all = tensor[:, 0, 0]           # (N, T, H, W)
    mems  = tensor[:, 1:, 0]          # (N, E-1, T, H, W)

    for i in range(N):
        y_t_hw = y_all[i]             # (T,H,W)
        preds_members_t_hw = mems[i]  # (E-1,T,H,W)
        update_running_stats(mse_sum, spread_sum, y_t_hw, preds_members_t_hw)

    mse = (mse_sum / N).numpy()
    spread = (spread_sum / N).numpy()
    return mse, spread


def compute_online_from_model(
    lit_module: LatentDiffusion,
    loader: DataLoader,
    device: torch.device,
    ensemble_size: int,
    var_index: int,
    lead_slice: slice
) -> Tuple[np.ndarray, np.ndarray]:
    """
    不保存大体积 .pt，在线逐样本生成集合成员并累计统计。
    """
    lit_module.eval()
    n_seen = 0
    mse_sum = None
    spread_sum = None

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            B = x_batch.shape[0]
            for sample_idx in range(B):
                x_single = x_batch[sample_idx: sample_idx + 1].to(device)
                y_single = y_batch[sample_idx: sample_idx + 1].to(device)

                # 取目标变量与 lead 切片 -> (T,H,W)
                # y: (1, C, T, H, W) -> 选 C=var_index, 再切时间
                y_t_hw = y_single[:, var_index, lead_slice, :, :].squeeze(0)  # (T,H,W)

                preds_members = []
                for ens_id in range(ensemble_size):
                    torch.manual_seed(ens_id + 1)  # 1..E，确保不同噪声
                    pred = lit_module.predict_step((x_single, None), batch_idx=0)
                    # pred: (1, C, T, H, W)
                    pred_t_hw = pred[:, var_index, lead_slice, :, :].squeeze(0)  # (T,H,W)
                    preds_members.append(pred_t_hw.cpu())

                    del pred
                    torch.cuda.empty_cache()

                preds_members_t_hw = torch.stack(preds_members, dim=0)         # (E, T, H, W)

                if mse_sum is None:
                    T = y_t_hw.shape[0]
                    mse_sum = torch.zeros(T)
                    spread_sum = torch.zeros(T)

                update_running_stats(mse_sum, spread_sum, y_t_hw.cpu(), preds_members_t_hw)

                n_seen += 1
                del x_single, y_single, preds_members_t_hw, preds_members, y_t_hw
                torch.cuda.empty_cache()
                gc.collect()

    mse = (mse_sum / max(n_seen, 1)).numpy()
    spread = (spread_sum / max(n_seen, 1)).numpy()
    return mse, spread


def plot_and_save(mse: np.ndarray, spread: np.ndarray, out_dir: Path, title: str = ""):
    out_dir.mkdir(parents=True, exist_ok=True)
    leads = np.arange(1, len(mse) + 1)

    plt.figure(figsize=(7.2, 4.4), dpi=140)
    plt.plot(leads, mse, label="Error variance (MSE)")
    plt.plot(leads, spread, label="Ensemble spread (variance)")
    plt.xlabel("Lead month t")
    plt.ylabel("Value")
    plt.title(title or "Spread–Skill (Error vs Spread) by Lead")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    png_path = out_dir / "spread_skill_curves.png"
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    # 另存 CSV
    csv_path = out_dir / "spread_skill_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lead_month", "mse_error_variance", "ensemble_spread_variance", "rmse", "sqrt_spread"])
        for t, (m, s) in enumerate(zip(mse, spread), start=1):
            w.writerow([t, m, s, np.sqrt(max(m, 0.0)), np.sqrt(max(s, 0.0))])

    print(f"[OK] Saved figure to: {png_path}")
    print(f"[OK] Saved metrics to: {csv_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lead_slice = parse_slice(args.lead_slice)

    # 1) 如果提供了已保存的 .pt，直接计算
    if args.from_saved_pt:
        print(f"Compute curves from saved: {args.from_saved_pt}")
        mse, spread = compute_from_saved_pt(args.from_saved_pt)
        plot_and_save(mse, spread, out_dir, title="From saved ensemble")
        return

    # 2) 否则：重建网络、载入权重、跑 DataModule，并在线统计
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        local_rank = 0

    device = torch.device(args.device)

    base_module, _ = setup_singleenc_model(
        future_timesteps=args.future_steps,
        local_rank=local_rank
    )

    lit_module: LatentDiffusion = LatentDiffusion.load_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=base_module.model,
        autoencoder=base_module.autoencoder,
        context_encoder=base_module.context_encoder,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        use_ema=True,
        parameterization="eps",
        map_location="cpu",
    ).to(device).eval()

    # 可选：若 LitModule 支持采样配置，这里尝试注入（静默失败）
    for attr, val in {
        "guidance_scale": None,  # 如需可加 CLI 并设置
        "ddim_eta": None,
        "sample_steps": None,
    }.items():
        if val is not None and hasattr(lit_module, attr):
            setattr(lit_module, attr, val)

    dm = SeaIceDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        future_steps=args.future_steps,
    )
    dm.setup(stage="test")

    loader = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()

    print(f"Starting online evaluation on split={args.split} …")
    mse, spread = compute_online_from_model(
        lit_module, loader, device,
        ensemble_size=args.ensemble_size,
        var_index=args.var_index,
        lead_slice=lead_slice
    )
    plot_and_save(mse, spread, out_dir, title=f"Online eval ({args.split})")

    # 清理
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
