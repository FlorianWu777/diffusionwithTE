import pytorch_lightning as pl
import torch
from ..diffusion.diffusion import SpreadMonitor
from ..diffusion import diffusion
import os


def setup_genforecast_training(
    model,
    autoencoder,
    context_encoder,
    model_dir,
    lr,
    num_gpus=None,            # ✅ 原有参数
    inference_mode=False,     # ✅ 原有参数
    cfg_dropout_p=0.4,        # ✅ 新增；默认 0.1，可忽略
    unconditional_guidance_scale=5,
    finetune=False,
    var_reg_weight=0.0
):
    """
    构建 LatentDiffusion LightningModule + Lightning Trainer

    Parameters
    ----------
    model, autoencoder, context_encoder : 与训练期一致的网络实例
    model_dir   : str | Path
        保存权重的目录。
    lr          : float
        学习率。
    num_gpus    : int | None
        每个进程可见的 GPU 数；None ⇒ 1。
    inference_mode : bool
        训练 = False；调参 / 快速验证 = True。
    cfg_dropout_p : float, optional
        条件-dropout 概率，用于 Classifier-Free Guidance 训练。
        旧代码不传此参数亦可正常运行。
    """
    # ---------- 1) LightningModule ----------
    ldm = diffusion.LatentDiffusion(
        model,
        autoencoder,
        context_encoder=context_encoder,
        lr=lr,
        cfg_dropout_p=cfg_dropout_p,   # ← 新功能；老 LatentDiffusion 会安全忽略
        unconditional_guidance_scale=unconditional_guidance_scale,
        finetune=finetune,
        var_reg_weight=0.0)

    # ---------- 2) 设备与并行策略 ----------
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if "LOCAL_WORLD_SIZE" in os.environ:
        devices = int(os.environ["LOCAL_WORLD_SIZE"])   # 由 torchrun 控制
    else:
        devices = num_gpus if num_gpus is not None else 1

    # 若只有 1 块 GPU/CPU，则自动降级为单进程策略
    strategy    = "ddp" if (isinstance(devices, int) and devices > 1) else "auto"

    # ---------- 3) 回调 ----------
    early_stopping = pl.callbacks.EarlyStopping(
        monitor      = "val_loss",
        patience     = 200,
        verbose      = True)
#        check_finite = False,   )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath        = model_dir,
        filename       = "{epoch}-{val_loss:.4f}",
        monitor        = "val_loss",
        save_top_k     = 10,
    )

    spread_cb = SpreadMonitor(K=4, log_every_n_steps=1)

    # ---------- 4) Trainer ----------
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,                  # ✅ 按 torchrun 自动对齐
        strategy=strategy,                # ✅ "ddp" 或 "auto"
        max_epochs=1000,
        callbacks=[early_stopping, checkpoint, spread_cb],
        inference_mode=inference_mode,
        # precision="16-mixed",  # 显存紧张可以打开
    )
    return ldm, trainer
