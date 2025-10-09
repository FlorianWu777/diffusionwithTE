#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_seaice_single_ensemble.py — Run inference with a trained single‑encoder
AFNO + 3D‑UNet sea‑ice diffusion model and build an *ensemble* of forecasts
(20 members by default) **for *every* sample in the validation and test splits.**

Usage (single GPU):
    python infer_seaice_single_ensemble.py \
        --checkpoint ./model/genforecast_single/lightning_logs/version_0/checkpoints/epoch=9-step=5000.ckpt \
        --data_root /share/wuhaotian/dataset1 \
        --out_dir ./inference_outputs \
        --save_netcdf 

Key changes vs. the original `infer_seaice_single.py`:
--------------------------------------------------------------------
* **--ensemble_size** CLI arg (default = 20).
* Iterate over *both* validation **and** test loaders, saving one file per
  sample:  `{split}_idx{global_idx:05d}.pt` (shape: [E, C, T, H, W]).
* Uses `torch.manual_seed(ens_idx)` to ensure each member gets a different
  noise realisation.
* Cleans up CUDA memory after each sample → safe on a single GPU.

The network architecture recreation is identical to training; only the
prediction loop changed.
"""

import argparse
import gc
import os
from pathlib import Path
from typing import List, Tuple

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, DistributedSampler
import torch.distributed as dist

# ======== 3rd‑party / project imports (unchanged) ========
from simpleautoencoder import ConvEncoder3D, ConvDecoder3D, AutoencoderKL
from models.genforecast import analysis, unet, training
from trainvqvae import ClimateForecastDataset
from models.diffusion.diffusion import LatentDiffusion

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run inference and create an ensemble for every sample in val & test splits."
    )
    p.add_argument("--checkpoint",type=str,default="/share/wuhaotian/diffusionDemo/model/genforecast_single/epoch=72-val_loss=0.2235.ckpt",
  #  p.add_argument("--checkpoint",type=str,default="/data/wuhaotian/diffusionDemo/model/genforecast_single/epoch=1-val_loss=0.2199.ckpt",
                   help="Path to .ckpt file produced by Lightning Trainer.")
    p.add_argument("--data_root",type=str,default="/data/wuhaotian/diffusionDemo/dataset1/",
                   help="Root directory with climate data (same layout as during training).")
    p.add_argument("--future_steps", type=int, default=12,
                   help="Number of lead months the model predicts.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for inference (e.g. cuda, cuda:1, cpu).")
    p.add_argument("--out_dir", default="/share/wuhaotian/inference_outputs",
                   help="Directory to save predictions.")
    p.add_argument("--save_netcdf", action="store_true",
                   help="Additionally save decoded predictions as NetCDF.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="Classifier‑Free guidance scale (higher → narrower spread).")
    p.add_argument("--eta", type=float, default=0.2,
                   help="DDIM eta noise (0 ⇒ deterministic).")
    p.add_argument("--steps", type=int, default=25,
                   help="Sampling steps; fewer steps + eta=0 → narrower spread.")
    p.add_argument("--ensemble_size", type=int, default=20,
                   help="Number of ensemble members per sample.")
    return p.parse_args()

# ---------------------- Data ----------------------

class SeaIceDataModule(pl.LightningDataModule):
    """Creates *train/val/test* loaders identical to training set‑up."""

    def __init__(self, root_dir: str, batch_size: int = 32, future_steps: int = 12):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.future_steps = future_steps

    # prepare_data left empty on purpose.

    def setup(self, stage: str | None = None):
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
            12, self.future_steps, "transfer", cmip_names, return_meta=True
        )
        self.reanal_dataset = ClimateForecastDataset(
            self.root_dir, variables, target_var,
            12, self.future_steps, "obs", return_meta=True
        )
        total_len = len(self.reanal_dataset)
        train_cutoff, valid_cutoff = total_len - 156, total_len - 120
        train_idx = list(range(0, train_cutoff))
        valid_idx = list(range(train_cutoff, valid_cutoff))
        test_idx = list(range(valid_cutoff, total_len))
        self.reanal_train_set = Subset(self.reanal_dataset, train_idx)
        self.reanal_valid_set = Subset(self.reanal_dataset, valid_idx)
        self.reanal_test_set = Subset(self.reanal_dataset, test_idx)

    # --- loaders ---
    def _dl(self, dataset, shuffle: bool, batch_size: int | None = None):
        sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
        return DataLoader(
            dataset, batch_size=batch_size or self.batch_size,
            sampler=sampler, num_workers=4, pin_memory=True, shuffle=False
        )

    def train_dataloader(self):
        return self._dl(self.cmip_dataset, shuffle=True)

    def val_dataloader(self):
        return self._dl(self.reanal_valid_set, shuffle=False)

    def test_dataloader(self):
        return self._dl(self.reanal_test_set, shuffle=False)

# ---------- Utility functions ----------

def setup_singleenc_model(  # *unchanged* architecture recreation
    past_timesteps: int = 12,
    future_timesteps: int = 12,
    autoenc_ckpt: str = "/data/wuhaotian/diffusionDemo/model/wae_finetune_best.pt",
    model_dir: str = "./model/genforecast_single",
    lr: float = 1e-4,
    local_rank: int = 0,
):
    world_size = torch.cuda.device_count() or 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    in_channels = 4          # 输入变量数量
    latent_channels = 32      # 降维后通道数
    kl_weight = 0.003
    
    # ---- 1. Autoencoder (frozen) ----
    encoder = ConvEncoder3D(in_ch=in_channels, fine_ch=latent_channels).to(device)
    decoder = ConvDecoder3D(fine_ch=latent_channels, out_ch=in_channels).to(device)
    
    autoenc = AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        kl_weight=kl_weight,        # ← 你原来就有
        kl_warmup_epochs=20         # ← 默认 20，也可以省略
    ).to(device)
#    autoencoder_mo
    autoenc.load_state_dict(torch.load(autoenc_ckpt, map_location="cpu"), strict=False)
    autoenc.to(device).eval()
    for p in autoenc.parameters():
        p.requires_grad_(False)

    # ---- 2. AFNO‑Cascade analysis net ----
    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoder=autoenc,
        embed_dim=64,
        analysis_depth=4,
        cascade_depth=3,
        input_patches=(1,),
        input_size_ratios=(1,),
        output_patches=future_timesteps,
        train_autoenc=True,
    )

    # ---- 3. 3D‑UNet in latent space ----
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

    # ---- 4. Lightning wrapper (inference mode) ----
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

# ---------------------- Ensemble prediction ----------------------

def generate_ensembles(
    lit_module: LatentDiffusion,
    loader: DataLoader,
    split_name: str,
    device: torch.device,
    ensemble_size: int,
    out_dir: Path,
):
    """Iterate over **every** sample in *loader* and save an ensemble tensor.

    Each output tensor has shape (E, C, T, H, W) and is stored under
    `{out_dir}/{split_name}_idx{global_idx:05d}.pt`.
    """

    global_idx = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    lit_module.eval()

    for batch_idx, batch in enumerate(loader):  # y_batch is the real values
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x_batch, y_batch, meta_batch = batch
        else:
            x_batch, y_batch = batch
            meta_batch = None
        B = x_batch.shape[0]
        for sample_idx in range(B):
            x_single = x_batch[sample_idx : sample_idx + 1].to(device)
            y_single = y_batch[sample_idx : sample_idx + 1].to(device)  # Get the real value y
            meta_single = None
            if isinstance(meta_batch, dict):
                meta_single = {
                    k: (v[sample_idx : sample_idx + 1].to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in meta_batch.items()
                }
            preds: List[torch.Tensor] = []

            # Add the true value y as the first ensemble member
            preds.append(y_single.squeeze(0).unsqueeze(0).cpu())  # (1, C, T, H, W)

            for ens_id in range(1, ensemble_size):  # Start from 1, since the first member is y
                torch.manual_seed(ens_id)  # new noise per member
                with torch.no_grad():
                    pred = lit_module.predict_step((x_single, None, meta_single), batch_idx=0)
                preds.append(pred.squeeze(0).unsqueeze(0).cpu())  # (1, C, T, H, W)

            ensemble_tensor = torch.cat(preds, dim=0)  # (E, C, T, H, W)
            save_path = out_dir / f"{split_name}_idx{global_idx:05d}.pt"
            torch.save(ensemble_tensor, save_path)
            print(f"✔ Saved {save_path}  →  {tuple(ensemble_tensor.shape)}")

            # cleanup
            del ensemble_tensor, preds, x_single, y_single, pred
            torch.cuda.empty_cache()
            gc.collect()

            global_idx += 1

# ---------------------- main ----------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- DDP / single‑GPU set‑up -----
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        local_rank = 0
    device = torch.device(args.device)

    # ----- (Re)create model architecture -----
    base_module, _ = setup_singleenc_model(future_timesteps=args.future_steps,
                                           local_rank=local_rank)

    # ----- Lightning module with checkpoint weights -----
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
        strict=False
    ).to(device).eval()

    # ----- DataModule -----
    dm = SeaIceDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        future_steps=args.future_steps,
    )
    dm.setup(stage="test")  # val & test are both created in setup()

    # ----- Predict & save ensembles -----
    print("Starting ensemble generation …")
  #  generate_ensembles(
   #     lit_module, dm.val_dataloader(), "val", device,
    #    args.ensemble_size, out_dir / "val"
    #)
    generate_ensembles(
        lit_module, dm.test_dataloader(), "test", device,
        args.ensemble_size, out_dir / "test"
    )
    print("All ensembles saved under:", out_dir)

    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
