#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_seaice_single_pretrained.py  —  从预训练模型继续训练的单 Encoder + AFNO + 3D‑UNet 季节海冰扩散模型

主要改动
-----------
1. **数据集**：改为使用 *reanalysis* 数据集的训练切片 (`self.reanal_train_set`) 作为训练集；验证/测试切片保持不变。
2. **加载预训练权重**：增加 `pretrained_ckpt` 实参；若路径存在，则在 LightningModule (`ldm`) 上加载权重后继续训练。
"""

import os
import gc
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler
import pytorch_lightning as pl

# ========= 你的实现 =========
from vaeblock        import SimpleConvEncoder, SimpleConvDecoder
from vaeautoencoder  import AutoencoderKL
from models.genforecast import analysis, unet, training
from dataloader      import ClimateForecastDataset
# ============================


class SeaIceDataModule(pl.LightningDataModule):
    """DataModule：使用 reanalysis 的 train/val/test 切片。"""

    def __init__(self, root_dir: str, batch_size: int = 4, future_steps: int = 3):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.future_steps = future_steps

    # LightningDataModule API
    # -----------------------
    def prepare_data(self):
        """无下载需求，占位。"""
        pass

    def setup(self, stage: str | None = None):
        variables  = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
        target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']

        # CMIP 数据集（保留作接口兼容）
        cmip_names = ['EC-Earth3/r2i1p1f1', 'MRI-ESM2-0/r1i1p1f1']
        self.cmip_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, 'transfer', cmip_names
        )

        # Reanalysis 数据集：划分 train / valid / test
        self.reanal_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, 'obs'
        )

        total_len = len(self.reanal_dataset)
        train_end = total_len - 156   # 与原脚本保持一致
        valid_end = total_len - 120

        # 索引切片
        train_idx = list(range(0, train_end))
        valid_idx = list(range(train_end, valid_end))
        test_idx  = list(range(valid_end, total_len))

        self.reanal_train_set = Subset(self.reanal_dataset, train_idx)
        self.reanal_valid_set = Subset(self.reanal_dataset, valid_idx)
        self.reanal_test_set  = Subset(self.reanal_dataset, test_idx)

    # ---- dataloaders ----
    def _dataloader(self, dataset, shuffle: bool):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(
            dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            shuffle     = (sampler is None) and shuffle,
            num_workers = 4,
            pin_memory  = True,
        )

    def train_dataloader(self):
        return self._dataloader(self.reanal_train_set, shuffle=True)

    def val_dataloader(self):
        return self._dataloader(self.reanal_valid_set, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.reanal_test_set, shuffle=False)


# ---------- 共用包装器 ----------
class SharedAE(torch.nn.Module):
    """让同一 Autoencoder 权重在 AFNO 里注册为多流。"""
    def __init__(self, ae):
        super().__init__()
        self.ae = ae

    def encode(self, x):
        return self.ae.encode(x)

    def forward(self, x):
        return self.ae(x)


# ---------- 模型构建 ----------

def setup_singleenc_model(
        past_timesteps: int  = 12,
        future_timesteps: int = 12,
        autoenc_ckpt: str    = "./model/autoencoder_best.pt",
        model_dir: str       = "./model/genforecast_single",
        lr: float            = 1e-4,
):
    """构建并返回 (LightningModule, Trainer)。"""

    world_size = torch.cuda.device_count()

    # 1️⃣ Autoencoder
    enc = SimpleConvEncoder(in_dim=4, levels=2, min_ch=64)
    dec = SimpleConvDecoder(in_dim=4, levels=2, min_ch=64)
    autoenc = AutoencoderKL(
        enc, dec,
        kl_weight        = 0.01,
        encoded_channels = 64,
        hidden_width     = 32,
    )
    autoenc.load_state_dict(torch.load(autoenc_ckpt, map_location="cpu"), strict=False)
    autoenc.eval()
    for p in autoenc.parameters():
        p.requires_grad_(False)

    # 2️⃣ AFNO‑Cascade（仅 obs 一条流）
    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoder        = autoenc,
        embed_dim          = 128,
        analysis_depth     = 4,
        cascade_depth      = 3,
        input_patches      = (1,),
        input_size_ratios  = (1,),
        output_patches     = future_timesteps // 4,
        train_autoenc      = False,
    )

    # 3️⃣ 3D‑UNet in latent space
    unet_model = unet.UNetModel(
        in_channels    = autoenc.hidden_width,  # 64
        out_channels   = autoenc.hidden_width,
        model_channels = 256,
        num_res_blocks = 2,
        attention_resolutions = (1, 2),
        dims           = 3,
        channel_mult   = (1, 2, 4),
        num_heads      = 4,
        num_timesteps  = future_timesteps,
        context_ch     = analysis_net.cascade_dims,
        use_checkpoint = True,
    )

    # 4️⃣ Lightning 封装
    # ⚠️ 注意：这里保持与库函数签名一致，**不要加关键字 unet_model=**
    ldm, trainer = training.setup_genforecast_training(
        unet_model,            # 第 1 个位置参数：UNet
        autoenc,               # 第 2 个位置参数：Autoencoder
        context_encoder = analysis_net,
        model_dir       = model_dir,
        lr              = lr,
        num_gpus        = world_size,
    )

    gc.collect()
    return ldm, trainer


# ---------- 训练入口 ----------

def train(
        root_dir: str        = "/share/wuhaotian/dataset1",
        batch_size: int      = 2,
        future_steps: int    = 12,
        pretrained_ckpt: str = "/data/wuhaotian/diffusionDemo/model/genforecast_single/epoch=13-val_loss_ema=0.0398.ckpt",
        model_dir: str       = "./model/genforecast_single",
):
    """主训练函数。若 `pretrained_ckpt` 存在，则加载后继续训练。"""

    # —— DDP 初始化 ——
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        local_rank = 0  # CPU 或单卡

    # —— Data ——
    datamodule = SeaIceDataModule(
        root_dir     = root_dir,
        batch_size   = batch_size,
        future_steps = future_steps,
    )
    datamodule.setup(stage="fit")

    # —— Model & Trainer ——
    model, trainer = setup_singleenc_model(future_timesteps=future_steps, model_dir=model_dir)

    # —— 读取预训练权重 ——
    if pretrained_ckpt and Path(pretrained_ckpt).is_file():
        ckpt = torch.load(pretrained_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"\n[Info] Loaded pretrained weights from {pretrained_ckpt}")
        if missing:
            print(f"Missing keys: {len(missing)} — {missing[:5]} …")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)} — {unexpected[:5]} …")

    # —— DataLoaders ——
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()

    # —— 训练 ——
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # —— 释放 ——
    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()


# ---------- 主函数 ----------
if __name__ == "__main__":
    train()
