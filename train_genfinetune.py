#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_seaice_single_finetune.py — 支持 torchrun --nproc_per_node=6
"""

import os, gc, torch
from pathlib import Path
from torch import distributed as dist
from torch.utils.data import DataLoader, Subset, DistributedSampler
import pytorch_lightning as pl

# ========= 你的实现 =========
from wae_train import ConvEncoder3D, ConvDecoder3D, AutoencoderWAE
from models.genforecast import analysis, unet, training
from trainvqvae import ClimateForecastDataset
# ============================


# --------------------------------------------------------------
#  DataModule
# --------------------------------------------------------------
class SeaIceDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=2, future_steps=12, finetune=False):
        super().__init__()
        self.root_dir      = root_dir
        self.batch_size    = batch_size
        self.future_steps  = future_steps
        self.finetune      = finetune

    def prepare_data(self): pass

    def setup(self, stage: str | None = None):
        variables   = ["psl/anom", "siconca/abs", "tas/anom", "tos/anom"]
        target_var  = variables

        cmip_runs = [
            "EC-Earth3/r2i1p1f1", "EC-Earth3/r7i1p1f1",
            "EC-Earth3/r10i1p1f1","EC-Earth3/r12i1p1f1",
            "EC-Earth3/r14i1p1f1",
            "MRI-ESM2-0/r1i1p1f1","MRI-ESM2-0/r2i1p1f1",
            "MRI-ESM2-0/r3i1p1f1","MRI-ESM2-0/r4i1p1f1",
            "MRI-ESM2-0/r5i1p1f1",
        ]
        self.cmip_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "transfer", cmip_runs
        )
        self.reanal_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "obs"
        )
        total_len     = len(self.reanal_dataset)
        train_cutoff  = total_len - 156
        valid_cutoff  = total_len - 120

        self.reanal_train_set = Subset(self.reanal_dataset, range(0, train_cutoff))
        self.reanal_valid_set = Subset(self.reanal_dataset, range(train_cutoff, valid_cutoff))
        self.reanal_test_set  = Subset(self.reanal_dataset, range(valid_cutoff, total_len))

    def _make_loader(self, dataset, shuffle, drop_last=False):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None) and shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=drop_last
        )

    def train_dataloader(self):
        if self.finetune:
            return self._make_loader(self.reanal_train_set, shuffle=True, drop_last=True)
        return self._make_loader(self.cmip_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):  return self._make_loader(self.reanal_valid_set, shuffle=False)
    def test_dataloader(self): return self._make_loader(self.reanal_test_set, shuffle=False)


# --------------------------------------------------------------
#  Model builder
# --------------------------------------------------------------
def setup_singleenc_model(
    future_timesteps=12,
    autoenc_ckpt="/data/wuhaotian/diffusionDemo/model/vae_finetune_best.pt",
    model_dir="/data/wuhaotian/diffusionDemo/model/genforecast_single",
    lr=1e-4,
    finetune_ckpt=None,
    finetune=True
):
    # DDP: 每个进程的 rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Autoencoder
    encoder = ConvEncoder3D(in_ch=4, fine_ch=32).to(device)
    decoder = ConvDecoder3D(fine_ch=32, out_ch=4).to(device)
    autoenc = AutoencoderWAE(encoder=encoder, decoder=decoder).to(device)

    if Path(autoenc_ckpt).is_file():
        sd = torch.load(autoenc_ckpt, map_location="cpu")
        autoenc.load_state_dict(sd, strict=False)

    for p in autoenc.encoder.parameters(): p.requires_grad_(False)
    for p in autoenc.decoder.parameters(): p.requires_grad_(True)

    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoder=autoenc,
        embed_dim=64,
        analysis_depth=4,
        cascade_depth=3,
        input_patches=(1,),
        input_size_ratios=(1,),
        output_patches=future_timesteps,
        train_autoenc=True
    )

    unet_model = unet.UNetModel(
        in_channels=autoenc.hidden_width,
        out_channels=autoenc.hidden_width,
        model_channels=128,
        num_res_blocks=2,
        attention_resolutions=(1,2),
        dims=3,
        channel_mult=(1,2,4),
        num_heads=4,
        num_timesteps=future_timesteps,
        context_ch=analysis_net.cascade_dims,
        use_checkpoint=True
    )

    ldm, trainer = training.setup_genforecast_training(
        unet_model,
        autoenc,
        context_encoder=analysis_net,
        model_dir=model_dir,
        lr=lr,
        cfg_dropout_p=0,
        unconditional_guidance_scale=1,
        finetune=finetune
    )
    return ldm, trainer


# --------------------------------------------------------------
#  Train entrypoint
# --------------------------------------------------------------
def train():
    from pytorch_lightning.callbacks import ModelCheckpoint
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--finetune_ckpt", type=str,
        default="/data/wuhaotian/diffusionDemo/model/epoch=epoch=32-val_loss=val_loss=0.0011.ckpt")
    args = parser.parse_args()

    # 初始化 DDP
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not dist.is_initialized():
            dist.init_process_group("nccl")

    datamodule = SeaIceDataModule(
        root_dir="/data/wuhaotian/diffusionDemo/dataset1",
        batch_size=args.batch_size,
        future_steps=12,
        finetune=True
    )
    datamodule.setup()

    model, trainer = setup_singleenc_model(
        future_timesteps=12,
        finetune_ckpt=args.finetune_ckpt,
        lr=1e-4,
        finetune=True
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath="./model",
        filename="epoch={epoch}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        save_top_k=-1,
        save_last=True
    )
    trainer.callbacks.append(checkpoint_cb)

    trainer.fit(model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
        ckpt_path=args.finetune_ckpt if Path(args.finetune_ckpt).is_file() else None
    )

    if dist.is_initialized():
        dist.destroy_process_group()
    gc.collect()


if __name__ == "__main__":
    train()
