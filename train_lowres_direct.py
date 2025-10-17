#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练直接在 1/4 分辨率下运行的海冰扩散模型。"""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from dataloader import ClimateForecastDataset, QuarterResolution
from models.direct.context import PastContextAdapter
from models.direct.unet import Simple3DUNet
from models.diffusion.direct_diffusion import DirectDiffusion


class SeaIceQuarterDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: Path, batch_size: int = 8, num_workers: int = 4, future_steps: int = 12):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.future_steps = future_steps
        self.transform = QuarterResolution(scale=0.5)

        self.variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
        self.target_var = ['siconca/abs']
        self.cmip_members = [
            'EC-Earth3/r2i1p1f1', 'EC-Earth3/r7i1p1f1', 'EC-Earth3/r10i1p1f1',
            'EC-Earth3/r12i1p1f1', 'EC-Earth3/r14i1p1f1',
            'MRI-ESM2-0/r1i1p1f1', 'MRI-ESM2-0/r2i1p1f1', 'MRI-ESM2-0/r3i1p1f1',
            'MRI-ESM2-0/r4i1p1f1', 'MRI-ESM2-0/r5i1p1f1'
        ]

    def setup(self, stage=None):
        self.train_dataset = ClimateForecastDataset(
            self.root_dir,
            self.variables,
            self.target_var,
            input_seq_len=12,
            output_seq_len=self.future_steps,
            mode='transfer',
            model_names=self.cmip_members,
            transform=self.transform,
        )

        obs_dataset = ClimateForecastDataset(
            self.root_dir,
            self.variables,
            self.target_var,
            input_seq_len=12,
            output_seq_len=self.future_steps,
            mode='obs',
            transform=self.transform,
        )
        total_len = len(obs_dataset)
        valid_start = total_len - 156
        test_start = total_len - 120

        self.valid_dataset = Subset(obs_dataset, list(range(valid_start, test_start)))
        self.test_dataset = Subset(obs_dataset, list(range(test_start, total_len)))

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset) if torch.distributed.is_available() and torch.distributed.is_initialized() else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        sampler = DistributedSampler(self.valid_dataset, shuffle=False) if torch.distributed.is_available() and torch.distributed.is_initialized() else None
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        sampler = DistributedSampler(self.test_dataset, shuffle=False) if torch.distributed.is_available() and torch.distributed.is_initialized() else None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def build_model(future_steps: int, variables: int, base_channels: int, context_channels: int) -> DirectDiffusion:
    context_adapter = PastContextAdapter(variables, context_channels, future_steps)
    unet = Simple3DUNet(
        in_channels=context_channels + 1,
        out_channels=1,
        base_channels=base_channels,
        depth=3,
    )
    return DirectDiffusion(
        model=unet,
        context_adapter=context_adapter,
        timesteps=1000,
        beta_schedule='cosine',
        lr=1e-4,
        cfg_dropout_p=0.2,
        future_steps=future_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=Path, required=True, help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--context_channels', type=int, default=32)
    parser.add_argument('--future_steps', type=int, default=12)
    args = parser.parse_args()

    pl.seed_everything(42)

    datamodule = SeaIceQuarterDataModule(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        future_steps=args.future_steps,
    )

    model = build_model(
        future_steps=args.future_steps,
        variables=len(datamodule.variables),
        base_channels=args.base_channels,
        context_channels=args.context_channels,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices='auto',
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
