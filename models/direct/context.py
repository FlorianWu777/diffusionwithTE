import torch
import torch.nn as nn
import torch.nn.functional as F


class PastContextAdapter(nn.Module):
    """将历史序列编码成与目标时间步对齐的条件特征。"""

    def __init__(self, in_channels: int, context_channels: int, future_steps: int):
        super().__init__()
        self.future_steps = future_steps
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, context_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(context_channels, context_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected input with 5 dims, got {x.shape}")
        if x.shape[2] != self.future_steps:
            x = F.interpolate(
                x,
                size=(self.future_steps, x.shape[-2], x.shape[-1]),
                mode="trilinear",
                align_corners=False,
            )
        return self.proj(x)
