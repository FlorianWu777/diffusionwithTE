"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
"""

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial
import torch.nn.functional as F
from .utils import make_beta_schedule, extract_into_tensor, noise_like, timestep_embedding
from .ema import LitEma
from ..blocks.afno import PatchEmbed3d, PatchExpand3d, AFNOBlock3d
from models.diffusion import ddim 
from pytorch_lightning.callbacks import Callback
from torchmetrics.regression import ContinuousRankedProbabilityScore
from .kcrps import KernelCRPS
from contextlib import nullcontext


def grad_map(x):                      # Sobel-like 3×3
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return torch.cat([gx, gy], dim=1)         # [B,2,T,H,W]

def laplacian(x):                     # 1-级 Laplacian pyramid
    low = F.avg_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2))
    up  = F.interpolate(low, scale_factor=(1,2,2), mode="trilinear", align_corners=False)
    return x - up

def neighborhood_filter(x, kernel_size=3):
    """
    应用一个简单的邻域滤波器，使用3x3的卷积核进行空间平滑
    x: [B, C, T, H, W]  ->  返回也为 [B, C, T, H, W]
    """
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device) / (kernel_size ** 2)
    kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size)

    # 将 [B, C, T, H, W] 视作 T 个 2D 卷积（按时间独立）
    B, C, T, H, W = x.shape
    x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)      # [B*T, C, H, W]

    x_filtered_reshaped = F.conv2d(x_reshaped, kernel, padding=padding, groups=C)

    # 还原为 [B, C, T, H, W]
    x_filtered = x_filtered_reshaped.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
    return x_filtered


class SpreadMonitor(Callback):
    """每 N 个验证 step 计算一次 ensemble spread 和 DPP 多样性指标。"""
    def __init__(self, K=4, log_every_n_steps=1, eta=0.01, steps=50):
        super().__init__()
        self.K = K
        self.log_every = log_every_n_steps
        self.eta = eta
        self.ddim_steps = steps

    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx % self.log_every != 0:
            return

        x, _,_ = batch  # x: [B, V, T, H, W]
        B, V, T, H, W = x.shape
        t_rel = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).repeat(B, 1)
        context = [(x, t_rel)]

        shape = (
            pl_module.autoencoder.hidden_width,
            pl_module.future_steps,
            H // 16,
            W // 16,
        )

        from models.diffusion import ddim
        sampler = ddim.DDIMSampler(pl_module, eta=self.eta)

        preds = []
        for _ in range(self.K):
            latents, _ = sampler.sample(
                self.ddim_steps,
                B,
                shape,
                conditioning=context,
                progbar=False,
            )
            y_hat = pl_module.autoencoder.decode(latents)   # [B, V, T, H, W]
            preds.append(y_hat)

        ens = torch.stack(preds, dim=0)     # [K,B,V,T,H,W]
        spread_map = ens.var(dim=0)         # [B,V,T,H,W]
        spread_scalar = spread_map.flatten(2).mean()

        # === DPP on Ensemble ===
        K, B, V, T, H, W = ens.shape
        ens_flat = ens.view(K, B, -1)  # [K, B, D]

        dpp_loss_vals = []
        for b in range(B):
            z = ens_flat[:, b, :]  # [K, D] - K samples for 1 condition
            dpp_val = dpp_loss(z)
            dpp_loss_vals.append(dpp_val)

        dpp_scalar = torch.stack(dpp_loss_vals).mean()

        # === Logging ===
        trainer.logger.log_metrics(
            {
                "val_spread": spread_scalar.item(),
                "val_dpp_ensemble": dpp_scalar.item(),
            },
            step=trainer.global_step,
        )

def compute_mmd(x, y, sigma=1.0):
    # x, y: [B, D]
    xx = torch.cdist(x, x, p=2)**2
    yy = torch.cdist(y, y, p=2)**2
    xy = torch.cdist(x, y, p=2)**2

    K_xx = torch.exp(-xx / (2 * sigma ** 2))
    K_yy = torch.exp(-yy / (2 * sigma ** 2))
    K_xy = torch.exp(-xy / (2 * sigma ** 2))

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd
    
def multi_bw_mmd2(z_p, z_q, sigmas=(1.0, 5.0, 10.0), weights=None):
    """
    z_p, z_q: [B, D] flattened latent vectors
    返回多带宽加权 MMD²
    """
    if weights is None:
        weights = [1 / len(sigmas)] * len(sigmas)

    mmd_total = 0.0
    for w, s in zip(weights, sigmas):
        mmd_total += w * compute_mmd(z_p, z_q, sigma=s)
    return mmd_total

def dpp_loss(z, eps=1e-5):
    """
    z: [B, D] 生成样本的特征矩阵
    """
    z = F.normalize(z, dim=1)  # 归一化以计算余弦相似度
    K = torch.matmul(z, z.T)   # [B, B] 相似度矩阵

    # 数值稳定处理
    K += eps * torch.eye(z.size(0), device=z.device)

    # 计算 log(det(K))，越大代表越 diverse
    sign, logdet = torch.slogdet(K)
    return -logdet if sign > 0 else torch.tensor(0.0, device=z.device)

    
class LatentDiffusion(pl.LightningModule):
    def __init__(self,
        model,
        autoencoder,
        context_encoder=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        use_ema=True,
        lr=1e-4,
        lr_warmup=0,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",  # all assuming fixed variance schedules
        future_steps = 12,
        unconditional_guidance_scale=1.0,
        cfg_dropout_p=0.2,
        stage2_start_step: int = 400000,   # ← N 个 global_step 后启用 Stage 2
        mmd_gamma_max: float = 0.3,       # 最终 γ
        dpp_lambda_max: float = 0.5,      # 最终 λ
        extra_noise_scale: float = 1.5,   # Stage 2 扩散噪声放大系数
        phys_mse_weight = 1.0,
        phys_crps_weight = 2.0,
        phys_brier_weight: float = 0.5, 
        finetune = False,
        var_reg_weight=0.0,
        # ===== DNCL 超参（新增） =====
        dncl_lambda_max: float = 0.7,     # DNCL 权重上限（配合 _ratio 使用，实际有效≈0.1*此值）
        dncl_start_step: int = 500000,      # DNCL 启动步（先学准，再拉开）
        dncl_warm_span: int = 1500000,      # DNCL 线性 warm-up 步数
        dncl_K: int = 3                    # DNCL 每步并行成员数（包含主样本），建议 2~4
    ):

        super().__init__()
        self.crps_metric = ContinuousRankedProbabilityScore()
        self.model = model
        self.autoencoder = autoencoder.requires_grad_(False)
        self.conditional = True
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.autoencoder.requires_grad_(False)
        self.finetune = finetune  
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.register_schedule(
            beta_schedule=beta_schedule, timesteps=timesteps,
            linear_start=linear_start, linear_end=linear_end, 
            cosine_s=cosine_s
        )

        self.loss_type = loss_type
        self.future_steps = future_steps
        self.cfg_dropout_p = cfg_dropout_p
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.stage2_start_step = stage2_start_step
        self.mmd_gamma_max = mmd_gamma_max
        self.dpp_lambda_max = dpp_lambda_max
        self.extra_noise_scale = extra_noise_scale
        self.var_reg_weight = float(var_reg_weight)
        # Pixel-space 额外损失的最大权重
        self.phys_mse_weight  = phys_mse_weight   # γ_max
        self.phys_crps_weight = phys_crps_weight  # λ_max

        # 物理域 CRPS（公平版即可）
        self.kcrps_loss = KernelCRPS(fair=True)
        self.phys_brier_weight = phys_brier_weight

        # ===== DNCL 配置（保存参数） =====
        self.dncl_lambda_max = float(dncl_lambda_max)
        self.dncl_start_step = int(dncl_start_step)
        self.dncl_warm_span  = int(dncl_warm_span)
        self.dncl_K          = int(max(1, dncl_K))

        # --- 供 Brier 使用：把物理阈值 0.15 映射到“反变换前”的标准化空间 ---
        # 你绘图时的反变换：phys = norm * 0.25055954 + 0.07842615
        # 因此 norm = (phys - shift)/scale
        scale = 0.25055954
        shift = 0.07842615
        thr_phys = 0.15
        thr_norm = (thr_phys - shift) / scale
        self.register_buffer('brier_thr_norm', torch.tensor(thr_norm, dtype=torch.float32))
        # ---- 额外两个缓冲，用来把 ε̂→x̂0 ----
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / self.alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / self.alphas_cumprod - 1.)))

        # [FIX] 新增：统一把预测还原到 x0 空间
    def _to_x0(self, x_t, t, model_out):
        if self.parameterization == "eps":
            return self.predict_start_from_noise(x_t, t, model_out)
        elif self.parameterization == "x0":
            return model_out
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")


    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        betas = make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start, linear_end=linear_end,
            cosine_s=cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas    = torch.tensor(betas, dtype=torch.float32) 
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod,  t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # stage helper
    def _in_stage2(self) -> bool:
        return self.global_step >= self.stage2_start_step
    
    # 线性 warm-up 到目标权重
    def _ratio(self, max_val, warm_span, start_step):
        if self.global_step < start_step:        # 未到 start_step => 0
            return 0.
        prog = (self.global_step - start_step) / warm_span
        return 0.1 * max_val * min(1.0, prog)    # 注意这里有 0.1 缩放
    
    def apply_model(self, x_noisy, t, cond=None, return_ids=False):
        """
        x_noisy: 当前扩散状态
        t      : 扩散步索引（保持原样，不要被 t_rel 覆盖）
        cond   : 期望为 [(x, t_rel)]；若误传为 [ [(x,t_rel)], [(0,t_rel)] ] 也能兼容
        """
        # 仅在 eval/验证阶段启用 EMA；训练阶段使用即时权重
        use_ema = (not self.training) and self.use_ema
        ctx_mgr = self.ema_scope() if use_ema else nullcontext()
    
        # 纯条件 / 纯无条件 或未启用CFG：直接一次前向
        if (self.unconditional_guidance_scale == 1.0) or (cond is None) or (not self.conditional):
            ctx = self.context_encoder(cond) if (self.conditional and cond is not None) else None
            with ctx_mgr:
                return self.model(x_noisy, t, context=ctx)
    
        # --------- CFG 分支 ---------
        # 统一出一个“基准上下文” base_cond = [(x, t_rel)]
        if isinstance(cond, list) and len(cond) == 2 and isinstance(cond[0], list):
            base_cond = cond[0]   # 兼容老的 [cond, uncond] 传参方式
        else:
            base_cond = cond
    
        # 构造无条件上下文：用同一 t_rel，但把 x 置零
        x_ctx, t_rel = base_cond[0]
        zero_cond = [(torch.zeros_like(x_ctx), t_rel)]
    
        # 编码 cond / uncond
        cond_ctx   = self.context_encoder(base_cond)
        uncond_ctx = self.context_encoder(zero_cond)
    
        # 两路前向；注意这里的 t 是扩散步索引
        with ctx_mgr:
            eps_c = self.model(x_noisy, t, context=cond_ctx)
            eps_u = self.model(x_noisy, t, context=uncond_ctx)
    
        s = float(self.unconditional_guidance_scale)
        return eps_u + s * (eps_c - eps_u)

    def apply_model_ori(self, x_noisy, t, cond=None, return_ids=False):
        if self.unconditional_guidance_scale == 1.0 or cond is None or not self.conditional:
            # Only conditional or unconditional
            if self.conditional and cond is not None:
                cond = self.context_encoder(cond)
            with self.ema_scope():
                return self.model(x_noisy, t, context=cond)
        else:
            # Classifier-Free Guidance
            cond_encoded = self.context_encoder(cond)
            x, t = cond[0]
            x_like = torch.zeros_like(x)
            uncond_encoded = [(x_like, t)]
    
            with self.ema_scope():
                cond_out = self.model(x_noisy, t, context=cond_encoded)
                uncond_out = self.model(x_noisy, t, context=uncond_encoded)
    
            return uncond_out + self.unconditional_guidance_scale * (cond_out - uncond_out)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True, use_smooth=False):
        if use_smooth:
            pred = neighborhood_filter(pred)
            target = neighborhood_filter(target)
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            return loss.mean() if mean else loss
        elif self.loss_type == 'l2':
            return F.mse_loss(target, pred, reduction='mean' if mean else 'none')
        else:
            raise NotImplementedError

    def p_losses(self, x_start, t, noise=None, context=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, context=context)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        # 使用邻域滤波后的目标和预测值计算损失
        return self.get_loss(model_out, target, mean=False).mean()

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def ori_shared_step(self, batch):
        x, y = batch
    
        # 编码目标 y（通道数固定）
        y = self.autoencoder.encode(y)[2] #should be 0, abolished dual coarse-fine 
        # 构造 context 输入：x -> [(x_i, t_relative), ...]
        B, V, T, H, W = x.shape
        t_relative = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).repeat(B, 1)  # shape [B, T]
        x_for_context = [(x, t_relative)]
        # 调用 context_encoder
        if self.cfg_dropout_p > 0:
            if isinstance(x_for_context, list):
                x_for_context_dropped = [
                    (c[0] if np.random.rand() > self.cfg_dropout_p else torch.zeros_like(c[0]), c[1])
                    for c in x_for_context
                ]
            else:
                x_for_context_dropped = x_for_context
        else:
            x_for_context_dropped = x_for_context
        
        context = self.context_encoder(x_for_context_dropped) if self.conditional else None
    
        return self(y, context=context)


    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss,on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        mse_loss = torch.tensor(0.0, device=self.device)
        ensemble_spread = torch.tensor(0.0, device=self.device)
        mse_mean_avg = torch.tensor(0.0, device=self.device)
        spread_mean_avg = torch.tensor(0.0, device=self.device)
        ssr_avg = torch.tensor(0.0, device=self.device)
    
        # 只处理第一个 batch 的第一个样本
        if batch_idx == 0:
            x, y,t_emb = batch  # x: [B, V, T, H, W], y: [B, C, T, H, W]
            x_single = x[0:1].to(self.device)
    
            ensemble_size = 20
            ensemble_preds = []
            for ens_id in range(ensemble_size):
                torch.manual_seed(ens_id)
                preds = self.predict_step((x_single, None), batch_idx=batch_idx)
                ensemble_preds.append(preds)
    
            # [E, C, T, H, W]
            ensemble_tensor = torch.cat(ensemble_preds, dim=0)
    
            # 原有的 per-member MSE（平均到成员），与成员方差的空间均值
            mse_list = []
            spread_list = []
    
            # 新增：ensemble mean 的 skill 与 spread，并计算 SSR
            mse_mean_list = []
            spread_mean_list = []
            ssr_list = []
    
            T = x_single.shape[2]
            for t in range(T):
                y_t = y[0, 1, t]                       # [H, W]
                preds_t = ensemble_tensor[:, 1, t]     # [E, H, W]
    
                # 原有统计：成员 MSE（与真值），成员间方差
                mse = F.mse_loss(preds_t, y_t.unsqueeze(0).expand(ensemble_size, -1, -1))
                spread = preds_t.var(dim=0)            # [H, W]
                mse_list.append(mse.item())
                spread_list.append(spread.mean().item())
    
                # 新增：ensemble mean 的 MSE（skill）与 spread
                ens_mean_t = preds_t.mean(dim=0)                  # [H, W]
                mse_mean = F.mse_loss(ens_mean_t, y_t)            # scalar
                spread_t = preds_t.var(dim=0)                     # [H, W]
                spread_mean = spread_t.mean()                     # scalar
                ssr = spread_mean / (mse_mean + 1e-8)             # spread-skill ratio
    
                mse_mean_list.append(mse_mean.item())
                spread_mean_list.append(spread_mean.item())
                ssr_list.append(ssr.item())
    
                # 打印两类统计
                print(f"Lead time = {t}, mse = {mse}, spread = {spread.mean()} ")
                print(f"Lead {t}: MSE(mean)={mse_mean:.4f}, spread={spread_mean:.4f}, SSR={ssr:.2f}")
    
            # 汇总平均
            mse_loss = float(np.mean(mse_list))
            ensemble_spread = float(np.mean(spread_list))
    
            mse_mean_avg = float(np.mean(mse_mean_list))
            spread_mean_avg = float(np.mean(spread_mean_list))
            ssr_avg = float(np.mean(ssr_list))
    
            print(f"Ensemble MSE loss at batch {batch_idx}: {mse_loss}")
            print(f"Ensemble spread (variance) at batch {batch_idx}: {ensemble_spread}")
            print(f"[AVG over T] MSE(mean)={mse_mean_avg:.4f}, spread={spread_mean_avg:.4f}, SSR={ssr_avg:.2f}")
    
            # 转成 tensor 便于 log（Lightning 期望 tensor/标量）
            mse_loss = torch.tensor(mse_loss, device=self.device)
            ensemble_spread = torch.tensor(ensemble_spread, device=self.device)
            mse_mean_avg = torch.tensor(mse_mean_avg, device=self.device)
            spread_mean_avg = torch.tensor(spread_mean_avg, device=self.device)
            ssr_avg = torch.tensor(ssr_avg, device=self.device)
    
        # 标准验证损失
        loss = self.shared_step(batch)
    
        # 记录日志
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mse_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ensemble_spread", ensemble_spread, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # 新增：ensemble mean 的 skill / spread / SSR
        self.log("val_mse_mean", mse_mean_avg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_spread_mean", spread_mean_avg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_ssr", ssr_avg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

        
    def shared_step_full(self, batch):
        x, y = batch                               # x: [B,V,T,H,W]
    
        # ===== latent =====
        y_latent = self.autoencoder.encode(y)[0]
        B, V, T, H, W = x.shape
    
        # ===== context =====
        t_rel   = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).repeat(B, 1)
        context = [(x if torch.rand(1).item() > self.cfg_dropout_p else torch.zeros_like(x), t_rel)]
        context_enc = self.context_encoder(context) if self.conditional else None
    
        # ===== diffusion：主任务 =====
        t        = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise    = torch.randn_like(y_latent)
        if self._in_stage2():                                   # 噪声渐升
            noise_scale = 1.0 + 0.5 * min(1.0, (self.global_step - self.stage2_start_step)/20000)
            noise = noise * noise_scale
    
        x_noisy   = self.q_sample(y_latent, t, noise)
        model_out = self.model(x_noisy, t, context=context_enc)
    
        # ===== latent recon loss =====
        target     = noise if self.parameterization == "eps" else y_latent
        loss_recon = self.get_loss(model_out, target).mean()
    
        # ===== DNCL：一致性（Teacher-Student, stop-grad, s<t, same ε）=====
        loss_dncl_cons = torch.zeros(1, device=self.device)
        lambda_dncl    = self._ratio(self.dncl_lambda_max, self.dncl_warm_span, self.dncl_start_step)
        if lambda_dncl > 0.0:
            # 采样 s < t：对每个样本独立、且 s 与 t 相关（避免 s >> t 造成尺度失配过大）
            # s = floor(u * t)，当 t=0 时令 s=0
            u = torch.rand_like(t, dtype=torch.float)
            s = (u * t.float()).long()
    
            # 关键：复用同一 ε
            eps_base = torch.randn_like(y_latent)
            x_t = self.q_sample(y_latent, t, eps_base)
            x_s = self.q_sample(y_latent, s, eps_base)
    
            # Student：当前权重
            out_t = self.model(x_t, t, context=context_enc)
            x0_hat_t = self._to_x0(x_t, t, out_t)
    
            # Teacher：EMA + stop-grad
            with torch.no_grad():
                with self.ema_scope():
                    out_s_tch = self.model(x_s, s, context=context_enc)
                x0_hat_s_tch = self._to_x0(x_s, s, out_s_tch).detach()
    
            # 归一化权重 w = 1/(σ_t² + σ_s²)
            sigma_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y_latent.shape)
            sigma_s = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, s, y_latent.shape)
            w = 1.0 / (sigma_t.pow(2) + sigma_s.pow(2) + 1e-6)
    
            loss_dncl_cons = (w * (x0_hat_t - x0_hat_s_tch).pow(2)).mean()
            loss_dncl_cons = lambda_dncl * loss_dncl_cons
    
        # ===== Stage-2 额外项（保持你的原逻辑）=====
        loss_phys_mse  = torch.zeros(1, device=self.device)
        loss_brier_sie = 5*torch.zeros(1, device=self.device)
    
        if self._in_stage2():
            latents_pred = (self.predict_start_from_noise(x_noisy, t, model_out)
                            if self.parameterization == "eps" else model_out)
    
            y_hat_v1  = self.autoencoder.decode(latents_pred)[:, 1]   # [B,T,H,W]
            target_v1 = y[:, 1]
    
            # ---- pixel-MSE γ ----
            gamma = self._ratio(self.phys_mse_weight, warm_span=10000,
                                start_step=self.stage2_start_step)
            loss_phys_mse = 10*gamma * F.mse_loss(y_hat_v1, target_v1)
    
        # ---------------- Brier（SIE）----------------
        lam_brier = self._ratio(self.phys_brier_weight, warm_span=8000,
                                start_step=self.stage2_start_step + 1000)
        if lam_brier > 0 and self._in_stage2():
            K = 4
            ens_members = []
            for _ in range(K):
                n_k = torch.randn_like(y_latent)
                xk  = self.q_sample(y_latent, t, n_k)
                out = self.model(xk, t, context=context_enc)
                lat = (self.predict_start_from_noise(xk, t, out)
                       if self.parameterization == "eps" else out)
                yk  = self.autoencoder.decode(lat)[:, 1]
                ens_members.append(yk)
    
            ens_stack = torch.stack(ens_members, dim=1)                    # [B,K,T,H,W]
            p_hat = (ens_stack > self.brier_thr_norm).float().mean(dim=1)  # \hat p
            y_bin = (y[:, 1] > self.brier_thr_norm).float()
            loss_brier_sie = lam_brier * F.mse_loss(p_hat, y_bin)
    
        # ===== total =====
        total_loss = 1*loss_recon + loss_phys_mse + 50*loss_brier_sie + loss_dncl_cons
    
        # ===== logging =====
        log_dict = {
            "loss_recon"       : loss_recon,
            "loss_phys_mse"    : loss_phys_mse,
            "loss_brier_sie"   : loss_brier_sie,
            "loss_dncl_cons"   : loss_dncl_cons,
            "lambda_dncl_eff"  : torch.as_tensor(lambda_dncl, device=self.device),
            "loss_total"       : total_loss,
        }
        if self.finetune:
            log_dict.update({"loss_hf_grad": torch.zeros(1, device=self.device),
                             "loss_hf_lap" : torch.zeros(1, device=self.device)})
    
        self.log_dict(log_dict, on_step=True, prog_bar=True, sync_dist=True)
        return total_loss
    def shared_step(self, batch):
        #if batch.len =3 
        x, y,t_emb = batch                               # x: [B,V,T,H,W]
        y_latent = self.autoencoder.encode(y,t_emb)[0]
        B, V, T, H, W = x.shape
    
        # ---- condition (per-sample dropout) ----
        t_rel = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).repeat(B, 1)
        keep  = (torch.rand(B,1,1,1,1, device=x.device) > self.cfg_dropout_p).float()
        x_cond = x * keep
        context = [(x_cond, t_rel)]
        context_enc = self.context_encoder(context) if self.conditional else None
    
        # ---- diffusion main ----
        t     = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(y_latent)
        if self._in_stage2():
            noise_scale = 1.0 + 0.5 * min(1.0, (self.global_step - self.stage2_start_step)/20000)
            noise = noise * noise_scale
    
        x_noisy   = self.q_sample(y_latent, t, noise)
        model_out = self.model(x_noisy, t, context=context_enc)
    
        # recon loss
        target     = noise if self.parameterization == "eps" else y_latent
        loss_recon = self.get_loss(model_out, target).mean()
    
        # ---- DNCL: reuse main forward on t ----
        loss_dncl_cons = x_noisy.new_zeros(())
        lambda_dncl    = self._ratio(self.dncl_lambda_max, self.dncl_warm_span, self.dncl_start_step)
        if lambda_dncl > 0.0:
            u = torch.rand_like(t, dtype=torch.float)
            s = (u * t.float()).long()
    
            eps_base = noise             # 复用主分支 ε
            x_t = x_noisy                # 复用主分支 x_t
            x_s = self.q_sample(y_latent, s, eps_base)
    
            out_t     = model_out
            x0_hat_t  = self._to_x0(x_t, t, out_t)
    
            with torch.no_grad():
                with self.ema_scope():
                    out_s_tch = self.model(x_s, s, context=context_enc)
                x0_hat_s_tch = self._to_x0(x_s, s, out_s_tch)
    
            sigma_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y_latent.shape)
            sigma_s = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, s, y_latent.shape)
            w = 1.0 / (sigma_t.pow(2) + sigma_s.pow(2) + 1e-6)
    
            loss_dncl_cons = (w * (x0_hat_t - x0_hat_s_tch).pow(2)).mean()
            loss_dncl_cons = lambda_dncl * loss_dncl_cons
    
        # ---- Stage-2 extra (按你原逻辑) ----
        loss_phys_mse  = x_noisy.new_zeros(())
        loss_brier_sie = x_noisy.new_zeros(()) * 5
    
        if self._in_stage2():
            latents_pred = (self.predict_start_from_noise(x_noisy, t, model_out)
                            if self.parameterization == "eps" else model_out)
            y_hat_v1  = self.autoencoder.decode(latents_pred)[:, 1]
            target_v1 = y[:, 1]
            gamma = self._ratio(self.phys_mse_weight, warm_span=10000,
                                start_step=self.stage2_start_step)
            loss_phys_mse = 10*gamma * F.mse_loss(y_hat_v1, target_v1)
    
            lam_brier = self._ratio(self.phys_brier_weight, warm_span=8000,
                                    start_step=self.stage2_start_step + 1000)
            if lam_brier > 0:
                K = 4
                ens_members = []
                for _ in range(K):
                    n_k = torch.randn_like(y_latent)
                    xk  = self.q_sample(y_latent, t, n_k)
                    out = self.model(xk, t, context=context_enc)
                    lat = (self.predict_start_from_noise(xk, t, out)
                           if self.parameterization == "eps" else out)
                    yk  = self.autoencoder.decode(lat)[:, 1]
                    ens_members.append(yk)
                ens_stack = torch.stack(ens_members, dim=1)
                p_hat = (ens_stack > self.brier_thr_norm).float().mean(dim=1)
                y_bin = (y[:, 1] > self.brier_thr_norm).float()
                loss_brier_sie = lam_brier * F.mse_loss(p_hat, y_bin)
    
        total_loss = loss_recon + loss_phys_mse + 10*loss_brier_sie + loss_dncl_cons
    
        self.log_dict({
            "loss_recon": loss_recon,
            "loss_phys_mse": loss_phys_mse,
            "loss_brier_sie": loss_brier_sie,
            "loss_dncl_cons": loss_dncl_cons,
            "lambda_dncl_eff": torch.as_tensor(lambda_dncl, device=self.device),
            "loss_total": total_loss,
        }, on_step=True, prog_bar=True, sync_dist=True)
    
        return total_loss


    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sampling_steps = 50
        x, _ = batch
        B, V, T, H, W = x.shape
    
        t_relative = torch.linspace(0, 1, T, device=x.device).unsqueeze(0).repeat(B, 1)
        context = [(x, t_relative)]
        shape = (self.autoencoder.hidden_width, self.future_steps, H // 16, W // 16)
    
        from models.diffusion import ddim
        sampler = ddim.DDIMSampler(self)
    
        # [FIX] 评估期：Stage1 用确定性（eta=0, temperature=1.0）
        eta = 0.0 if not self._in_stage2() else 0.05
        temperature = 1.0  # [FIX] 标量温度，避免被广播成逐步噪声表
    
        if self.unconditional_guidance_scale == 1.0:
            context_input = context
        else:
            zero_context = [(torch.zeros_like(x), t_relative)]
            context_input = [context, zero_context]
    
        latents, _ = sampler.sample(
            sampling_steps, B, shape,
            conditioning=context_input,
            eta=eta,
            progbar=False,
            temperature=temperature,    # [FIX] 使用标量
        )
        return self.autoencoder.decode(latents)

