#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 在文件顶部加上：
from models.genforecast.unet import UNetModel
import torch.nn.functional as F

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
from models.diffusion.decoder_finetune import DecoderFineTuneWithSampler
from models.diffusion.plms import PLMSSampler
from models.diffusion import diffusion
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
            num_workers= 4,
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
        autoenc_ckpt: str = "/data/wuhaotian/diffusionDemo/model/simpleautoencoder_scaledftbest.pt",
        model_dir: str = "./model/genforecast_single",
        lr: float = 1e-5,
        local_rank: int = 0 
):
    world_size = torch.cuda.device_count()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1) 单 Encoder / Decoder / Autoencoder
# 使用你的 3D 卷积 + Attention 自定义结构
    enc = ConvEncoder3D(in_channels=4, latent_channels=32)  # latent dim 可调
    dec = ConvDecoder3D(latent_channels=32, out_channels=4)
    autoenc = AutoencoderKL(
        encoder=enc,
        decoder=dec,
        kl_weight=0.01,
        encoded_channels=32,
        hidden_width=32,
    )
    autoenc.load_state_dict(torch.load(autoenc_ckpt, map_location="cpu"), strict=False)
#    autoenc.to(device)
 #   autoenc.eval()
  #  for p in autoenc.parameters():
   #     p.requires_grad_(False)
    for p in autoenc.encoder.parameters():
        p.requires_grad_(False)
    if hasattr(autoenc, "quant_conv"):
        for p in autoenc.quant_conv.parameters():
            p.requires_grad_(False)
    if hasattr(autoenc, "post_quant_conv"):
        for p in autoenc.post_quant_conv.parameters():
            p.requires_grad_(False)
    # 如果 KL 里还有 learnable logvar，也一并冻结
    if hasattr(autoenc, "logvar"):
        autoenc.logvar.requires_grad_(False)
    
    # 2️⃣  只保留 Decoder 可训练
    for p in autoenc.decoder.parameters():
        p.requires_grad_(True)
    

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
    # 从 checkpoint 恢复模型
    ldm = diffusion.LatentDiffusion.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        autoencoder=autoencoder,
        context_encoder=context_encoder,
        lr=lr,
        cfg_dropout_p=0.1,  # 必须加上你 `__init__` 里定义的所有参数
        unconditional_guidance_scale=1
    )

        # 3️⃣  冻结 UNet（潜空间去噪）和 A F N O context-encoder
    for p in unet_model.parameters():
        p.requires_grad_(False)
    for p in analysis_net.parameters():
        p.requires_grad_(False)
    # 4) Lightning 封装
    _, trainer = training.setup_genforecast_training(
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

# ---------- loading ----------
def build_components(args):
    # ---------- AutoEncoder ----------
    ae = AutoencoderKL(
        ConvEncoder3D(4,32), ConvDecoder3D(32,4),
        kl_weight=0.01, encoded_channels=32, hidden_width=32
    )
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location="cpu"), strict=False)

    #   · 冻结 encoder & KL 相关
    for n,p in ae.named_parameters():
        if not n.startswith("decoder"): p.requires_grad_(False)

    # ---------- Analysis / AFNO ----------
    analysis_net = analysis.AFNONowcastNetCascade(
        autoencoder = ae, embed_dim=64, analysis_depth=4, cascade_depth=3,
        input_patches=(1,), input_size_ratios=(1,), output_patches=args.future_steps,
        train_autoenc=False
    )

    # ---------- UNet ----------
    # 1. 构造 UNet
    unet_model = unet.UNetModel(
        in_channels=ae.hidden_width, out_channels=ae.hidden_width,
        model_channels=128, num_res_blocks=2, attention_resolutions=(1,2),
        dims=3, channel_mult=(1,2,4), num_heads=4,
        num_timesteps=args.future_steps, context_ch=analysis_net.cascade_dims,
        use_checkpoint=True
    )
    
    # 2. 加载 UNet 和 context_encoder 的参数
    if args.ldm_ckpt:
        ckpt = torch.load(args.ldm_ckpt, map_location="cpu")["state_dict"]
        unet_sd = {k.split("unet.")[1]: v for k, v in ckpt.items() if k.startswith("unet.")}
        unet_model.load_state_dict(unet_sd, strict=False)
    
        ctx_sd  = {k.split("context_encoder.")[1]: v for k, v in ckpt.items() if k.startswith("context_encoder.")}
        analysis_net.load_state_dict(ctx_sd, strict=False)
    
    # 3. 构造 scheduler & 注入属性
    class SimpleLinearScheduler:
        def __init__(self, num_timesteps=1000):
            self.betas = torch.linspace(0.0001, 0.02, steps=num_timesteps)
            alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
    
    scheduler = SimpleLinearScheduler(num_timesteps=1000)
    
    # ✅ 关键：使用 unet 命名空间下的类方法
    from models.genforecast.unet import attach_plms_attrs

    unet_model = attach_plms_attrs(unet_model, scheduler)

    # 冻结 UNet & Analysis
    for p in unet_model.parameters():   p.requires_grad_(False)
    for p in analysis_net.parameters(): p.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = ae.to(device)
    unet_model = unet_model.to(device)
    analysis_net = analysis_net.to(device)
    return ae, unet_model, analysis_net

# ---------- 训练入口 ----------
import argparse
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str,
                   default="/data/wuhaotian/diffusionDemo/dataset1/",
                   help="Root folder that contains your ClimateForecastDataset files")

    p.add_argument("--ae_ckpt", type=str,
                   default="/data/wuhaotian/diffusionDemo/model/finetuned_autoencoder.pt",
                   help="Path to pre-trained AutoEncoder (only decoder will be finetuned)")

    p.add_argument("--ldm_ckpt", type=str,
                   default="/data/wuhaotian/diffusionDemo/model/genforecast_single/epoch=31-val_loss_ema=0.0297.ckpt",
                   help="Lightning checkpoint that stores UNet + context encoder weights")

    p.add_argument("--finetune_ckpt", type=str,
                   default="/data/wuhaotian/diffusionDemo/model/finetuned_autoencoder.pt",
                   help="Optional: Path to finetuned decoder weights")

    p.add_argument("--finetune",default=False, action="store_true",
                   help="Whether to finetune decoder on reanalysis data")

    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--future_steps", type=int, default=12)
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--max_epochs", type=int, default=30)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.gpus)))

    # 构建组件
    autoenc, unet_model, analysis_net = build_components(args)

    # 如果有 finetune_ckpt，加载 decoder 参数
    if args.finetune_ckpt:
        decoder_state = torch.load(args.finetune_ckpt, map_location="cpu")
        autoenc.load_state_dict(decoder_state, strict=False)
        print("✅ Loaded finetuned decoder weights from:", args.finetune_ckpt)

    # 构建 Lightning 模型
    finetuner = DecoderFineTuneWithSampler(
        autoenc=autoenc,
        unet=unet_model,
        context_encoder=analysis_net,
        lr=1e-4,
        lpips_w=0.05,
        num_sample_steps=25
    )

    # 构建数据模块（根据是否finetune来选数据集）
    dm = SeaIceDataModule(args.root_dir, args.batch_size,
                          args.future_steps, finetune=args.finetune)
    dm.setup()
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/topk",             # 存储目录
        filename="topk-{epoch:02d}-{val_loss_ema:.4f}",  # 文件名格式
        monitor="val_loss_ema",                   # 监控指标
        mode="min",                               # 越小越好
        save_top_k=10,                            # 保存 top 10
        save_last=True,                           # 同时保存最新模型
        auto_insert_metric_name=False
    )
    # Trainer
    trainer = pl.Trainer(
    accelerator="gpu", devices=args.gpus,
    precision=32, max_epochs=args.max_epochs,
    log_every_n_steps=20,
    default_root_dir="./logs/dec_ft",
    callbacks=[checkpoint_callback]
      )

    trainer.fit(finetuner,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders  =dm.val_dataloader())

    # 如果是 finetune 模式则保存

    Path("./model").mkdir(exist_ok=True)
    torch.save(autoenc.state_dict(), "./model/finetuned_autoencoder.pt")
    print("✅ Finetuned decoder saved to ./model/finetuned_autoencoder.pt")


# ---------- 主函数 ----------
if __name__ == "__main__":
    main()
