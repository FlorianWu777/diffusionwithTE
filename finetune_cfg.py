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
from trainvqvae      import ClimateForecastDataset
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
        
# ---------- 共用包装器 ----------
class SharedAE(torch.nn.Module):

    def __init__(self, ae): super().__init__(); self.ae = ae
    def encode(self, x):   return self.ae.encode(x)
    def forward(self, x):  return self.ae(x)

# ---------- 模型构建 ----------
def setup_singleenc_model(
        past_timesteps: int = 12,
        future_timesteps: int = 12,
        autoenc_ckpt: str = "/data/wuhaotian/diffusionDemo/model/ae_finetune_best.pt",
        model_dir: str = "./model/genforecast_single",
        lr: float = 1e-5,
        local_rank: int = 0 
):
    world_size = torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1) 单 Encoder / Decoder / Autoencoder
# 使用你的 3D 卷积 + Attention 自定义结构
    in_channels = 4          # 输入变量数量
    latent_channels = 32      # 降维后通道数
    kl_weight = 0.003
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    encoder = ConvEncoder3D(in_ch=in_channels, fine_ch=latent_channels).to(device)
    decoder = ConvDecoder3D(fine_ch=latent_channels, out_ch=in_channels).to(device)
    
    autoenc = AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        kl_weight=kl_weight,        # ← 你原来就有
        kl_warmup_epochs=20         # ← 默认 20，也可以省略
    ).to(device)
    autoenc.load_state_dict(torch.load(autoenc_ckpt, map_location="cpu"), strict=False)
    autoenc.to(device)
    autoenc.eval()
    for p in autoenc.parameters():
        p.requires_grad_(False)

    # 2) AFNO-Cascade (只有 obs 一条流；如要 NWP 再添一条)
    ae_list          = autoenc
    input_patches    = [1]      # ? 保持 list
    input_ratios     = [1]      # ? 保持 list   ← 关键!
    embed_dim        = 64
    analysis_depth   = 4
    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoder        = autoenc,      # 关键字最好写出来更直观
        embed_dim          = 64,          # 标量 OK；或 (128,)
        analysis_depth     = 4,            # 标量
        cascade_depth = 3,
        input_patches      = (1,),         # tuple
        input_size_ratios  = (1,),         # tuple
        output_patches     = future_timesteps,
        train_autoenc      = True
    )

    # 3) 3D-UNet in latent space
    unet_model = unet.UNetModel(
        in_channels   = autoenc.hidden_width,   #
        out_channels  = autoenc.hidden_width,
        model_channels= 128,
        num_res_blocks= 2,
        attention_resolutions=(1, 2),
        dims          = 3,
        channel_mult  = (1, 2, 4),
        num_heads     = 4,
        num_timesteps = future_timesteps,
        context_ch    = analysis_net.cascade_dims,
        use_checkpoint = True
    )

    # 4) Lightning 封装
    ldm, trainer = training.setup_genforecast_training(
        unet_model,
        autoenc,
        context_encoder = analysis_net,
        model_dir       = model_dir,
        lr              = lr,
        num_gpus      = world_size,
        cfg_dropout_p=0.15
    )
    gc.collect()
    return ldm, trainer

# ---------- DataModule ----------

# ---------- 训练入口 ----------
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument(
        "--resume_ckpt",
        default="/data/wuhaotian/diffusionDemo/model/genforecast_single/epoch=32-val_loss_ema=0.0213.ckpt",
        help="checkpoint to resume / fine-tune from",
    )

    args = parser.parse_args()

    # -------- 自动检测可用 GPU 并设置环境变量 --------
    all_gpus = torch.cuda.device_count()

    # 安全限制在实际设备数范围内
    if args.num_gpus > all_gpus:
        raise ValueError(f"Requested {args.num_gpus} GPUs, but only {all_gpus} are available.")
    
    gpu_ids = list(range(args.num_gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    available_gpus = args.num_gpus  # ? 设置为实际要用的 GPU 数量
    print(f"[INFO] Using {available_gpus} GPUs: {gpu_ids}")


    torch.backends.cudnn.benchmark = True

    # -------- DDP 初始化 --------
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    
        # 获取当前 rank 应该使用的 GPU id
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_list = list(map(int, visible_devices.split(","))) if visible_devices else list(range(torch.cuda.device_count()))
        if local_rank >= len(gpu_list):
            raise RuntimeError(f"Invalid LOCAL_RANK={local_rank}, but only {len(gpu_list)} GPUs exposed.")
    
        torch.cuda.set_device(gpu_list[local_rank])  # ? 安全分配实际 GPU
        dist.init_process_group("nccl")
    else:
        local_rank = 0


    # -------- 数据模块 --------
    datamodule = SeaIceDataModule(
        root_dir="/data/wuhaotian/diffusionDemo/dataset1",
        batch_size=args.batch_size,
        future_steps=12,
        finetune=True
    )
    datamodule.setup(stage="fit")

    # -------- 模型 --------
    model, trainer = setup_singleenc_model(local_rank=local_rank)  # ? 使用已有 Trainer


    # -------- Trainer 自动配置 --------
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        ckpt_path=args.resume_ckpt,       # ← 只加这一行
    )

    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()

# ---------- 主函数 ----------
if __name__ == "__main__":
    train()
