#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, gc, torch
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
from pathlib import Path
from lightning.pytorch.core.datamodule import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
# ========= 你的实现 =========
from simpleautoencoder        import ConvEncoder3D, ConvDecoder3D, AutoencoderKL
from models.genforecast import analysis, unet, training
# from dataloader      import ClimateForecastDataset
from trainvqvae import ClimateForecastDataset
# ============================

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class SeaIceDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=32, future_steps=3, finetune=False):
        super().__init__()
        self.root_dir      = root_dir
        self.batch_size    = batch_size
        self.future_steps  = future_steps
        self.finetune      = finetune

    # 无需下载 → 留空
    def prepare_data(self):
        pass

    def setup(self, stage: str | None = None):
        variables   = ["psl/anom", "siconca/abs", "tas/anom", "tos/anom"]
        target_var  = variables

        # 1) CMIP (pre-train)
        cmip_runs = [
            "EC-Earth3/r2i1p1f1",  "EC-Earth3/r7i1p1f1",
            "EC-Earth3/r10i1p1f1", "EC-Earth3/r12i1p1f1",
            "EC-Earth3/r14i1p1f1",
            "MRI-ESM2-0/r1i1p1f1", "MRI-ESM2-0/r2i1p1f1",
            "MRI-ESM2-0/r3i1p1f1", "MRI-ESM2-0/r4i1p1f1",
            "MRI-ESM2-0/r5i1p1f1",
        ]
        self.cmip_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "transfer", cmip_runs
        )

        # 2) Reanalysis (fine-tune / val / test)
        self.reanal_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "obs"
        )

        total_len     = len(self.reanal_dataset)
        train_cutoff  = total_len - 156
        valid_cutoff  = total_len - 120

        train_idx = list(range(0, train_cutoff))
        valid_idx = list(range(train_cutoff, valid_cutoff))
        test_idx  = list(range(valid_cutoff, total_len))

        self.reanal_train_set = Subset(self.reanal_dataset, train_idx)
        self.reanal_valid_set = Subset(self.reanal_dataset, valid_idx)
        self.reanal_test_set  = Subset(self.reanal_dataset, test_idx)

    # -------------- loaders --------------
    def _make_loader(self, dataset, shuffle, drop_last=False):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle    = (sampler is None) and shuffle,
            sampler    = sampler,
            num_workers= 2,
            pin_memory = True,
            drop_last  = drop_last,
        )

    def train_dataloader(self):
        if self.finetune:
            return self._make_loader(self.reanal_train_set, shuffle=True, drop_last=True)
        return self._make_loader(self.cmip_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._make_loader(self.reanal_valid_set, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.reanal_test_set, shuffle=False)
        


# ---------- DataModule ----------

# ---------- 训练入口 ----------


def train():


    # -------- 数据模块 --------
    datamodule = SeaIceDataModule(
        root_dir="/data/wuhaotian/diffusionDemo/dataset1",
        batch_size=64,
        future_steps=12,
        finetune=False
    )
    datamodule.setup() 
        # 取一个 batch（不打乱的前几个样本）
    first_batch = next(iter(datamodule.train_dataloader()))  # 返回 (x, y)
    x_batch, _ = first_batch  # x: (B, C, T, H, W)
    
    # 打印每个样本、每个通道的 max, min, std
    B, C, T, H, W = x_batch.shape
    x_np = x_batch.numpy()
    
    for b in range(B):
        print(f"\nSample {b}:")
        for c in range(C):
            sample_channel = x_np[b, c]  # shape: (T, H, W)
            max_val = sample_channel.max()
            min_val = sample_channel.min()
            std_val = sample_channel.std()
            print(f"  Channel {c}: max={max_val:.4f}, min={min_val:.4f}, std={std_val:.4f}")



# ---------- 主函数 ----------
if __name__ == "__main__":
    train()
