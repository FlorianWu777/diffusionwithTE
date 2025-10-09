import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np


def kl_from_standard_normal(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # 0.5 * (exp(logσ²) + μ² − 1 − logσ²)
    return 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var).mean()


def ensemble_nll_normal(ensemble: torch.Tensor, sample: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    mean = ensemble.mean(dim=1)                                # [B, C, T, H, W]
    var  = ensemble.var(dim=1, unbiased=True) + epsilon        # [B, C, T, H, W]
    logvar = var.log()

    diff = sample[:, None, ...] - mean                         # [B, 1, C, T, H, W]
    logtwopi = np.log(2.0 * np.pi)
    # NLL = 0.5 * (log(2πσ²) + (x-μ)²/σ²)
    nll = 0.5 * (logtwopi + logvar + diff.square() / var).mean()
    return nll


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        kl_weight: float = 0.01,
        encoded_channels: int = 64,
        hidden_width: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width

        # 将 encoder 的特征映射到均值与对数方差
        self.to_moments = nn.Conv3d(encoded_channels, 2 * hidden_width, kernel_size=1)
        # 将隐变量映回到 decoder 的通道维
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels, kernel_size=1)

        self.kl_weight = kl_weight
        # 如果不使用，全局 log_var 参数可以删除；保留不影响功能
        # self.log_var = nn.Parameter(torch.zeros(size=()))

    @staticmethod
    def sample_from_standard_normal(mean: torch.Tensor, log_var: torch.Tensor, num: int | None = None, training: bool = True) -> torch.Tensor:
        """
        Reparameterization trick:
          z = mean + exp(0.5 * log_var) * eps,  eps ~ N(0, I)
        - training=True  : 采样
        - training=False : 不采样，等价 eps=0
        """
        std = (0.5 * log_var).exp()
        if num is not None:
            # 扩一个样本维（如需多样本采样）
            mean = mean[:, None, ...]
            std  = std[:,  None, ...]
        eps = torch.randn_like(mean) if training else torch.zeros_like(mean)
        return mean + std * eps

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mean, log_var = torch.chunk(self.to_moments(h), 2, dim=1)   # 各为 [B, hidden_width, T, H, W]
        return mean, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input: torch.Tensor, sample_posterior: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encode(input)
        if sample_posterior:
            # 根据当前模块是否处于训练模式决定是否真正采样
            z = self.sample_from_standard_normal(mean, log_var, training=self.training)
        else:
            z = mean
        dec = self.decode(z)
        return dec, mean, log_var

    def _loss(self, batch):
        (x, y) = batch
        # 若你的 DataModule 返回更复杂的结构，可在此处按需展开
        while isinstance(x, (list, tuple)):
            x = x[0][0]

        y_pred, mean, log_var = self.forward(x)
        rec_loss = (y - y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)
        total_loss = rec_loss + self.kl_weight * kl_loss
        return total_loss, rec_loss, kl_loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split: str = "val"):
        total_loss, rec_loss, kl_loss = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}_loss", total_loss, **log_params)
        self.log(f"{split}_rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}_kl_loss", kl_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.25, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_rec_loss",  # 本类里会记录这个 key
                "frequency": 1,
            },
        }
