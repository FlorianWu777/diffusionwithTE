"""Lightning module implementing conditional diffusion in pixel space."""

from __future__ import annotations

import inspect
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DirectConditionalDiffusion(pl.LightningModule):
    """Denoising diffusion model that operates directly on pixel-space tensors."""

    def __init__(
        self,
        network: nn.Module,
        target_channels: int,
        cond_channels: int,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")

        self.save_hyperparameters(ignore=["network"])
        self.network = network
        self.target_channels = target_channels
        self.cond_channels = cond_channels

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

        forward_params = list(inspect.signature(self.network.forward).parameters.values())
        # drop "self"
        if forward_params:
            forward_params = forward_params[1:]
        self._timesteps_param = None
        self._forward_accepts_kwargs = False
        for param in forward_params:
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                self._forward_accepts_kwargs = True
            if param.name in {"timesteps", "t", "time", "time_steps"}:
                self._timesteps_param = param
                break

    @property
    def num_timesteps(self) -> int:
        return int(self.betas.shape[0])

    def forward(
        self,
        noisy_target: torch.Tensor,
        cond: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise component from a noisy target."""

        t_channel = self._time_channel(timesteps, noisy_target)
        model_input = torch.cat([noisy_target, cond, t_channel], dim=1)
        if self._timesteps_param is not None:
            if self._timesteps_param.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }:
                if self._timesteps_param.kind is inspect.Parameter.POSITIONAL_ONLY:
                    return self.network(model_input, timesteps)
                if self._timesteps_param.default is inspect._empty:
                    return self.network(model_input, timesteps)
            return self.network(model_input, **{self._timesteps_param.name: timesteps})
        if self._forward_accepts_kwargs:
            return self.network(model_input, timesteps=timesteps)
        return self.network(model_input)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        cond, target = self._prepare_batch(batch)
        loss = self._diffusion_loss(cond, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=target.size(0))
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        cond, target = self._prepare_batch(batch)
        loss = self._diffusion_loss(cond, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=target.size(0))

    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            cond, target = batch[0], batch[1]
        else:
            raise ValueError("Batch must contain conditioning and target tensors")

        if isinstance(cond, (list, tuple)):
            cond = cond[0]
        if isinstance(target, (list, tuple)):
            target = target[0]

        cond = cond.float()
        target = target.float()

        cond = self._flatten_spatiotemporal(cond)
        target = self._flatten_spatiotemporal(target)
        if cond.shape[1] != self.cond_channels:
            raise RuntimeError(
                f"Conditioning channels mismatch: expected {self.cond_channels}, got {cond.shape[1]}"
            )
        if target.shape[1] != self.target_channels:
            raise RuntimeError(
                f"Target channels mismatch: expected {self.target_channels}, got {target.shape[1]}"
            )
        return cond, target

    def _flatten_spatiotemporal(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 5:
            raise ValueError("Expected tensor shaped [batch, variables, timesteps, H, W]")
        tensor = tensor.contiguous()
        b, v, t, h, w = tensor.shape
        return tensor.reshape(b, v * t, h, w)

    def _diffusion_loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = target.size(0)
        device = target.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(target)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noisy_target = sqrt_alpha * target + sqrt_one_minus * noise

        pred_noise = self.forward(noisy_target, cond, t)
        return F.mse_loss(pred_noise, noise)

    def _time_channel(self, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        h, w = reference.shape[-2:]
        t = timesteps.float() / max(1, self.num_timesteps - 1)
        channel = t.view(-1, 1, 1, 1).expand(-1, 1, h, w)
        return channel.to(reference.dtype).to(reference.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


__all__ = ["DirectConditionalDiffusion"]
