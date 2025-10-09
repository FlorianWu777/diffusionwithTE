#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
forecast_demo.py — 参照原作者的 `Forecast` 类，把 **海冰季节预测 LDM** 打包成即用接口，
并提供单机 / 多 GPU 分布式两种调用方式。

用法示例（单卡或自动选 GPU）：
```bash
python forecast_demo.py \
       --autoenc_ckpt   ./model/autoencoder_best.pt \
       --ldm_ckpt       ./model/genforecast_single/epoch=13-val_loss_ema=0.0398.ckpt \
       --root           /share/wuhaotian/dataset1 \
       --nsamples       2 \
       --out            ./vis_samples
```
```bash
# 多 GPU 并行取样（自动按卡数 spawn 子进程）
python forecast_demo.py --distributed \
       --autoenc_ckpt ./model/autoencoder_best.pt \
       --ldm_ckpt     ./model/genforecast_single/epoch=13-val_loss_ema=0.0398.ckpt \
       --root         /share/wuhaotian/dataset1
```

· **不做降水对数变换**，只对张量做 `float32 → torch.Tensor` 与设备搬运。  
· 输入形状固定 `(T_hist=12, C=4, H, W)`；输出 `(T_future=12, C=4, H, W)`。
"""

import argparse, os, gc, contextlib, multiprocessing as mp
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from finetune_gencast import setup_singleenc_model, SeaIceDataModule
from models.diffusion import plms  # PLMS 采样器

# -----------------------------------------------------------
# Forecast (single GPU)
# -----------------------------------------------------------
class Forecast:
    def __init__(self, *, ldm_ckpt, autoenc_ckpt, past_timesteps=12,
                 future_timesteps=12, gpu='auto'):
        self.past_timesteps   = past_timesteps
        self.future_timesteps = future_timesteps

        # ---- 构建 LDM（与训练完全一致） ----
        self.ldm, _ = setup_singleenc_model(
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            autoenc_ckpt=autoenc_ckpt,
            model_dir='tmp_dir')
        ckpt = torch.load(ldm_ckpt, map_location='cpu')
        self.ldm.load_state_dict(ckpt['state_dict'], strict=False)
        self.ldm.eval()

        # ---- 设备 ----
        if gpu is None:
            self.device = torch.device('cpu')
        else:
            if gpu == 'auto':
                gpu = 0 if torch.cuda.device_count() > 0 else None
            self.device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
            self.ldm.to(self.device)

        # ---- 采样器 ----
        self.sampler = plms.PLMSSampler(self.ldm)
        gc.collect()

    # ---------------- predict ----------------
    @torch.no_grad()
    def __call__(self, x_hist: np.ndarray, num_diffusion_iters: int = 50):
        """x_hist shape: (T_hist, C, H, W)，已归一化"""
        assert x_hist.shape[0] == self.past_timesteps
    
        # -------- 0. 预处理 --------
        x = torch.from_numpy(x_hist).unsqueeze(0).float().to(self.device)  # (1,T,C,H,W)
        x = x.permute(0, 2, 1, 3, 4)                                       # (1,C,T,H,W)
    
        # -------- 1. context --------
        B, C, T_hist, H, W = x.shape
        t_relative = torch.linspace(0, 1, T_hist, device=self.device).unsqueeze(0)
        cond = [(x, t_relative)]                                           # 列表，元素是 (tensor, time)
    
        # -------- 2. 采样 latent --------
        future_patches = self.future_timesteps // 4        # autoenc_time_ratio = 4
        gen_shape = (
            self.ldm.autoencoder.hidden_width,             # 32
            future_patches,                                # T_future / 4
            H // 4, W // 4                                 # 空间同样 /4
        )
        lat_out, _ = self.sampler.sample(
            num_diffusion_iters, B, gen_shape, cond, progbar=False
        )                                                  # (B,32,T_future/4,h,w)
    
        # -------- 3. 解码 --------
        B, Ch, T_fut, h, w = lat_out.shape          # Ch = 32
      #   lat_flat = lat_out.reshape(-1, Ch, h, w)    # (B*T_fut, 32, h, w)   ← 直接 reshape!
        lat_flat = lat_out 
        y_flat = self.ldm.autoencoder.decode(lat_flat)  # (B*T_fut, 4, H, W)
        
        # y_pred = y_flat.reshape(B, T_fut, 4, H, W) \
         #                .permute(0, 2, 1, 3, 4)        # (B, 4, T_fut, H, W)
        y_pred = y_flat
        return y_pred.squeeze(0).cpu().numpy()             # (C,T_future,H,W)

# -----------------------------------------------------------
# ForecastDistributed (multi‑GPU)
# -----------------------------------------------------------
class ForecastDistributed:
    def __init__(self, **cfg):
        self.cfg = cfg
        ctx = mp.get_context('spawn')
        self.in_q  = ctx.Queue()
        self.out_q = ctx.Queue()
        self.nprocs = max(1, torch.cuda.device_count())
        self.workers = mp.spawn(_worker, args=(self.in_q,self.out_q,cfg), nprocs=self.nprocs, join=False)
        for _ in range(self.nprocs):
            self.out_q.get()  # 等待就绪

    def __call__(self, x_batch: np.ndarray, num_diffusion_iters=50):
        N = x_batch.shape[0]
        for i in range(N):
            self.in_q.put((x_batch[i], num_diffusion_iters, i))
        preds = np.empty((N, self.cfg['future_timesteps']) + x_batch.shape[2:], dtype=x_batch.dtype)
        remaining = N
        while remaining:
            y, idx = self.out_q.get()
            preds[idx] = y
            remaining -= 1
        return preds

    def __del__(self):
        for _ in range(self.nprocs):
            self.in_q.put(None)
        self.workers.join()

# ---- worker proc ----
def _worker(rank, in_q, out_q, cfg):
    gpu = rank if torch.cuda.device_count()>0 else None
    fc = Forecast(ldm_ckpt=cfg['ldm_ckpt'], autoenc_ckpt=cfg['autoenc_ckpt'],
                  past_timesteps=cfg['past_timesteps'], future_timesteps=cfg['future_timesteps'], gpu=gpu)
    out_q.put('ready')
    while (msg:=in_q.get()) is not None:
        x, iters, idx = msg
        y = fc(x, num_diffusion_iters=iters)
        out_q.put((y, idx))

# -----------------------------------------------------------
# Quick demo: load a few samples from dataset & visualize
# -----------------------------------------------------------

def quick_demo(args):
    fc = Forecast(ldm_ckpt=args.ldm_ckpt, autoenc_ckpt=args.autoenc_ckpt,
                  past_timesteps=12, future_timesteps=12)

    dm = SeaIceDataModule(root_dir=args.root, batch_size=1, future_steps=12)
    dm.setup('fit')
    loader = dm.train_dataloader()

    os.makedirs(args.out, exist_ok=True)
    var_names = ['psl','siconca','tas','tos']

    for i,(x_hist,y_true) in enumerate(loader):
        if i>=args.nsamples: break
        x_np = x_hist.squeeze(0).permute(1, 0, 2, 3).numpy()

        y_pred = fc(x_np)                 # (C,Tfut,H,W)
        # ----- 简单可视化第一通道 -----
        plt.figure(figsize=(12,3))
        plt.subplot(131); plt.imshow(y_true[0,1,0].numpy()); plt.title('True t=0 ch1'); plt.axis('off')
        plt.subplot(132); plt.imshow(y_pred[1,0]); plt.title('Pred t=0 ch1'); plt.axis('off')
        diff = y_true[0,1,0].numpy()-y_pred[1,0]
        plt.subplot(133); plt.imshow(diff,cmap='bwr'); plt.title('Diff'); plt.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(args.out,f'sample{i}.png')); plt.close()
        print(f'saved sample {i}')

# -----------------------------------------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--autoenc_ckpt', required=True)
    p.add_argument('--ldm_ckpt', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--nsamples', type=int, default=48)
    p.add_argument('--out', default='./vis_samples')
    args = p.parse_args()

    if args.distributed:
        print('Distributed forecast not yet demo‑wired; run single GPU demo instead.')
    quick_demo(args)
