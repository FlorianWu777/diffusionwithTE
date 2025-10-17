import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Optional

from .utils import make_beta_schedule, extract_into_tensor
from .ddim import DDIMSampler


class DirectDiffusion(pl.LightningModule):
    """在降采样后的像素空间直接训练的扩散模型。"""

    def __init__(
        self,
        model: nn.Module,
        context_adapter: Optional[nn.Module] = None,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l2",
        lr: float = 1e-4,
        cfg_dropout_p: float = 0.1,
        parameterization: str = "eps",
        future_steps: int = 12,
    ):
        super().__init__()
        self.model = model
        self.context_adapter = context_adapter
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.lr = lr
        self.cfg_dropout_p = cfg_dropout_p
        self.parameterization = parameterization
        self.future_steps = future_steps

        self.register_schedule(timesteps=timesteps, beta_schedule=beta_schedule)

    def register_schedule(self, timesteps: int, beta_schedule: str):
        betas = make_beta_schedule(beta_schedule, timesteps)
        betas = torch.tensor(betas, dtype=torch.float32)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", torch.cat([torch.tensor([1.0], dtype=torch.float32), self.alphas_cumprod[:-1]]))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_variance", betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.num_timesteps = timesteps

    def _split_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, _ = batch
        elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            raise ValueError("Expected batch to be (x, y) or (x, y, meta)")
        return x, y

    def _encode_condition(self, x: torch.Tensor, apply_dropout: bool) -> Optional[torch.Tensor]:
        if self.context_adapter is None:
            return None
        cond = self.context_adapter(x)
        if cond.shape[2] != self.future_steps:
            cond = F.interpolate(
                cond,
                size=(self.future_steps, cond.shape[-2], cond.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        if apply_dropout and self.cfg_dropout_p > 0:
            keep = (torch.rand(cond.shape[0], device=cond.device) > self.cfg_dropout_p).float()
            keep = keep.view(-1, 1, 1, 1, 1)
            cond = cond * keep
        return cond

    def _concat_inputs(self, noisy: torch.Tensor, cond: Optional[torch.Tensor]):
        if cond is None:
            return noisy
        if cond.shape[2] != noisy.shape[2]:
            cond = F.interpolate(
                cond,
                size=(noisy.shape[2], noisy.shape[-2], noisy.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return torch.cat([noisy, cond], dim=1)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none")
        return F.mse_loss(pred, target, reduction="none")

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None):
        model_input = self._concat_inputs(x, cond)
        return self.model(model_input, t)

    def apply_model(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None):
        return self.forward(x, t, cond)

    def training_step(self, batch, batch_idx):
        x, y = self._split_batch(batch)
        x = x.to(self.device)
        y = y.to(self.device)
        cond = self._encode_condition(x, apply_dropout=True)

        B = y.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(y)
        noisy = self.q_sample(y, t, noise)
        pred = self.apply_model(noisy, t, cond)
        target = noise if self.parameterization == "eps" else y
        loss = self.get_loss(pred, target).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self._split_batch(batch)
        x = x.to(self.device)
        y = y.to(self.device)
        cond = self._encode_condition(x, apply_dropout=False)

        B = y.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(y)
        noisy = self.q_sample(y, t, noise)
        pred = self.apply_model(noisy, t, cond)
        target = noise if self.parameterization == "eps" else y
        loss = self.get_loss(pred, target).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def predict_future(self, x: torch.Tensor, steps: int = 50, eta: float = 0.0) -> torch.Tensor:
        self.eval()
        cond = self._encode_condition(x.to(self.device), apply_dropout=False)
        if cond is not None:
            spatial = cond.shape[-2:]
        else:
            spatial = x.shape[-2:]
        shape = (self.model.out_channels if hasattr(self.model, "out_channels") else x.shape[1], self.future_steps, *spatial)
        sampler = DDIMSampler(self)
        samples, _ = sampler.sample(steps, x.shape[0], shape, conditioning=cond, eta=eta, progbar=False)
        return samples

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = self._split_batch(batch)
        return self.predict_future(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
