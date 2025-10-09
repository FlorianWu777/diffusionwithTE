#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, gc, torch
import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from lightning.pytorch.core.datamodule import LightningDataModule
from models.genforecast.analysis import AFNONowcastNetCascade, AFNONowcastNetCascade_raw

# ========= 你的实现 =========
#from simpleautoencoder import ConvEncoder3D, ConvDecoder3D, AutoencoderKL
#from wae_train import ConvEncoder3D, ConvDecoder3D, AutoencoderWAE
from simpleautoencoder import ConvEncoder3D, ConvDecoder3D
from simpleautoencoder import  AutoencoderKL 
from models.genforecast import analysis, unet, training
from trainvqvae import ClimateForecastDataset
from annual_embedding import YearAdditiveEmbedding
from train_vae_embed import AEWithTE
# ============================


# ---------------- DataModule ----------------
class SeaIceDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=16, future_steps=12, finetune=False):
        super().__init__()
        self.root_dir      = root_dir
        self.batch_size    = batch_size     # ← 每卡 batch
        self.future_steps  = future_steps
        self.finetune      = finetune

    def prepare_data(self): pass

    def setup(self, stage: str | None = None):
        variables   = ["psl/anom", "siconca/abs", "tas/anom", "tos/anom"]
        target_var  = variables

        # 1) CMIP (pre-train)
        cmip_runs = [
            "EC-Earth3/r2i1p1f1","EC-Earth3/r7i1p1f1","EC-Earth3/r10i1p1f1",
            "EC-Earth3/r12i1p1f1","EC-Earth3/r14i1p1f1",
            "MRI-ESM2-0/r1i1p1f1","MRI-ESM2-0/r2i1p1f1","MRI-ESM2-0/r3i1p1f1",
            "MRI-ESM2-0/r4i1p1f1","MRI-ESM2-0/r5i1p1f1",
        ]
        self.cmip_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "transfer", cmip_runs,return_meta=True
        )

        # 2) Reanalysis (fine-tune / val / test)
        self.reanal_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "obs",return_meta=True
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

    def _make_loader(self, dataset, shuffle, drop_last=False):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(
            dataset,
            batch_size = self.batch_size,                 # 每卡 batch
            shuffle    = (sampler is None) and shuffle,
            sampler    = sampler,
            num_workers= 6,
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


# ---------------- 包装器（可选） ----------------
class SharedAE(torch.nn.Module):
    def __init__(self, ae): super().__init__(); self.ae = ae
    def encode(self, x):   return self.ae.encode(x)
    def forward(self, x):  return self.ae(x)


# ---------------- 模型构建 ----------------
def setup_singleenc_model(
        past_timesteps: int = 12,
        future_timesteps: int = 12,
        autoenc_ckpt: str = "/data/wuhaotian/diffusionDemo/model/vae_finetune_best.pt",
        model_dir: str = "./model/genforecast_single",
        lr: float = 3e-4,
        local_rank: int = 0,
        num_gpus: int = 8,                    # 固定 6 卡
        train_autoenc: bool = False,           # 冻结 AE 做编码器（默认 False）
        var_reg_weight=0.25 
):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1) AutoEncoder (VAE-KL)
    in_channels    = 4
    latent_channels= 32
    kl_weight      = 0.003

    encoder = ConvEncoder3D(in_ch=in_channels, fine_ch=latent_channels).to(device)
    decoder = ConvDecoder3D(fine_ch=latent_channels, out_ch=in_channels).to(device)

    ae_core = AutoencoderKL(
        encoder=encoder,
        decoder=decoder).to(device)
      #  kl_weight=kl_weight,
       # kl_warmup_epochs=20
    
    te = YearAdditiveEmbedding(num_channels=4, init_zero=True).to(device)
    autoenc = AEWithTE(ae_core, te).to(device)
    # 载入 ckpt 并按需冻结
    sd = torch.load(autoenc_ckpt, map_location="cpu")
    autoenc.load_state_dict(sd, strict=False)
    autoenc.to(device).eval()
    if not train_autoenc:
        for p in autoenc.parameters():
            p.requires_grad_(False)

    # 2) AFNO-Cascade
    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoder        = autoenc,
        embed_dim          = 64,
        analysis_depth     = 4,
        cascade_depth      = 3,
        input_patches      = (1,),
        input_size_ratios  = (1,),
        output_patches     = future_timesteps,
        train_autoenc      = train_autoenc
    )
    # afno_raw_model = AFNONowcastNetCascade_raw(
    #    autoencoder        = autoenc,
    #     embed_dim          = 64,
    #     analysis_depth     = 4,
    #     cascade_depth      = 3,
    #     input_patches      = (1,),
    #     input_size_ratios  = (1,),
    #     output_patches     = future_timesteps,
    #     train_autoenc      = train_autoenc
    # ).to(device)

    # # 2. 实例化时间嵌入（TE）模块
    # te_model = YearAdditiveEmbedding(num_channels=4, init_zero=True).to(device)

    # # 3. 实例化“包装器”，将上面两个组件组装起来
    # #    这才是你最终要用于训练的完整模型
    # analysis_net = AFNONowcastNetCascade(
    #     afno_model=afno_raw_model, 
    #     te_model=te_model
    # ).to(device)

    # 3) 3D-UNet in latent space
    unet_model = unet.UNetModel(
        in_channels   = 32,      # 与 AE latent 匹配
        out_channels  = 32,
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

    # 4) Lightning 封装（固定 6 卡）
    ldm, trainer = training.setup_genforecast_training(
        unet_model,
        autoenc,
        context_encoder = analysis_net,
        model_dir       = model_dir,
        lr              = lr,
        num_gpus        = num_gpus,     # ← 固定传 6
        cfg_dropout_p   = 0.05,
        unconditional_guidance_scale = 8.0,
        var_reg_weight = var_reg_weight
    )

    gc.collect()
    return ldm, trainer


# ---------------- 训练入口 ----------------
import argparse

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=12)     # 每卡 batch
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--num_gpus", type=int, default=8)       # 固定 6
    parser.add_argument("--train_autoenc", default=False)
    args = parser.parse_args()

    # 至少 6 卡
    avail = torch.cuda.device_count()
    if avail < 6:
        raise RuntimeError(f"Need 6 GPUs, but only {avail} detected.")

    torch.backends.cudnn.benchmark = True

    # ---- DDP 初始化（用 torchrun） ----
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        # 仍允许单进程调试，但目标是 6 卡
        local_rank = 0
    # ---- DataModule（每卡 batch） ----
    datamodule = SeaIceDataModule(
        root_dir="/data/wuhaotian/diffusionDemo/dataset1",
        batch_size=args.batch_size,     # 每卡
        future_steps=12,
        finetune=False
    )
    datamodule.setup(stage="fit")

    # ---- 模型 & Trainer ----
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath="./model",
        filename="epoch={epoch}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        save_top_k=5,
        save_last=True
    )

    model, trainer = setup_singleenc_model(
        local_rank=local_rank,
        num_gpus=6,
        train_autoenc=args.train_autoenc
    )
    trainer.callbacks.append(checkpoint_cb)

    # ---- 训练 ----
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader())
      #  ckpt_path="/data/wuhaotian/diffusionDemo/model/epoch=epoch=78-val_loss=val_loss=0.2126.ckpt")

    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()


if __name__ == "__main__":
    train()
