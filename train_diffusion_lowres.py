#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train a low-resolution diffusion model directly in pixel space."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from models.direct import DirectConditionalDiffusion, Simple3DUNet
from trainvqvae import ClimateForecastDataset


class SeaIceLowResDataModule(pl.LightningDataModule):
    """DataModule that wraps ClimateForecastDataset with spatial downsampling."""

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 8,
        future_steps: int = 12,
        past_steps: int = 12,
        scale_factor: float = 0.25,
        downsample_mode: str = "bilinear",
        finetune: bool = False,
        num_workers: int = 6,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.future_steps = future_steps
        self.past_steps = past_steps
        self.scale_factor = scale_factor
        self.downsample_mode = downsample_mode
        self.finetune = finetune
        self.num_workers = num_workers

        self.cond_channels: Optional[int] = None
        self.target_channels: Optional[int] = None
        self.lowres_shape: Optional[Sequence[int]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        variables = ["psl/anom", "siconca/abs", "tas/anom", "tos/anom"]
        target_var = variables

        cmip_runs = [
            "EC-Earth3/r2i1p1f1",
            "EC-Earth3/r7i1p1f1",
            "EC-Earth3/r10i1p1f1",
            "EC-Earth3/r12i1p1f1",
            "EC-Earth3/r14i1p1f1",
            "MRI-ESM2-0/r1i1p1f1",
            "MRI-ESM2-0/r2i1p1f1",
            "MRI-ESM2-0/r3i1p1f1",
            "MRI-ESM2-0/r4i1p1f1",
            "MRI-ESM2-0/r5i1p1f1",
        ]

        dataset_kwargs = dict(
            root_dir=self.root_dir,
            variables=variables,
            target_var=target_var,
            input_seq_len=self.past_steps,
            output_seq_len=self.future_steps,
            spatial_downsample=self.scale_factor,
            downsample_mode=self.downsample_mode,
            return_meta=False,
        )

        self.cmip_dataset = ClimateForecastDataset(
            mode="transfer",
            model_names=cmip_runs,
            **dataset_kwargs,
        )

        self.reanal_dataset = ClimateForecastDataset(
            mode="obs",
            **dataset_kwargs,
        )

        total_len = len(self.reanal_dataset)
        train_cutoff = total_len - 156
        valid_cutoff = total_len - 120

        train_idx = list(range(0, train_cutoff))
        valid_idx = list(range(train_cutoff, valid_cutoff))
        test_idx = list(range(valid_cutoff, total_len))

        self.reanal_train_set = Subset(self.reanal_dataset, train_idx)
        self.reanal_valid_set = Subset(self.reanal_dataset, valid_idx)
        self.reanal_test_set = Subset(self.reanal_dataset, test_idx)

        sample_x, sample_y = self.cmip_dataset[0]
        self.cond_channels = sample_x.shape[0] * sample_x.shape[1]
        self.target_channels = sample_y.shape[0] * sample_y.shape[1]
        self.lowres_shape = sample_y.shape[-2:]

    def _make_loader(self, dataset, shuffle: bool, drop_last: bool = False):
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None) and shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        dataset = self.reanal_train_set if self.finetune else self.cmip_dataset
        return self._make_loader(dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self._make_loader(self.reanal_valid_set, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.reanal_test_set, shuffle=False)


def build_model(datamodule: SeaIceLowResDataModule, args: argparse.Namespace) -> DirectConditionalDiffusion:
    if datamodule.cond_channels is None or datamodule.target_channels is None:
        raise RuntimeError("DataModule must be set up before building the model")

    channel_mult = tuple(int(x.strip()) for x in args.channel_mult.split(",") if x.strip())
    if not channel_mult:
        raise ValueError("channel_mult must specify at least one multiplier")
    network = Simple3DUNet(
        in_channels=datamodule.target_channels + datamodule.cond_channels + 1,
        out_channels=datamodule.target_channels,
        base_channels=args.base_channels,
        channel_multipliers=channel_mult,
        norm_groups=args.norm_groups,
    )

    return DirectConditionalDiffusion(
        network=network,
        target_channels=datamodule.target_channels,
        cond_channels=datamodule.cond_channels,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a low-resolution diffusion model")
    parser.add_argument("--root_dir", type=str, default="/data/wuhaotian/diffusionDemo/dataset1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--future_steps", type=int, default=12)
    parser.add_argument("--past_steps", type=int, default=12)
    parser.add_argument("--scale_factor", type=float, default=0.25)
    parser.add_argument("--downsample_mode", type=str, default="bilinear")
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--channel_mult", type=str, default="1,2,4")
    parser.add_argument("--norm_groups", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--default_root_dir", type=str, default="./model/lowres_direct")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--num_workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    datamodule = SeaIceLowResDataModule(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        future_steps=args.future_steps,
        past_steps=args.past_steps,
        scale_factor=args.scale_factor,
        downsample_mode=args.downsample_mode,
        finetune=args.finetune,
        num_workers=args.num_workers,
    )
    datamodule.setup(stage="fit")

    model = build_model(datamodule, args)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch}-val_loss={val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.devices if accelerator == "gpu" else 1
    strategy = "ddp" if accelerator == "gpu" and devices > 1 else None

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.grad_clip,
        precision=args.precision,
        default_root_dir=args.default_root_dir,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[checkpoint_cb, lr_monitor],
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
