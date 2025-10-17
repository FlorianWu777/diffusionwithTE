import torch
import torch.nn as nn

class YearAdditiveEmbedding(nn.Module):
    """
    Learnable additive bias for annual cycle.
    weight shape: [12, V]  (month 1..12, channel-wise)
    Add to x: [B, V, T, H, W].
    """
    def __init__(self, num_channels: int, init_zero: bool = True):
        super().__init__()
        self.weight = nn.Embedding(12, num_channels)  # [12, V]
        if init_zero:
            nn.init.zeros_(self.weight.weight)        # 从0开始，不扰动初始数据

    @torch.no_grad()
    def month_from_relative(self, B: int, T: int, device):
        # 如果拿不到绝对月份，用窗口内相对位相近似到 12 桶
        t_rel = torch.linspace(0, 1, T, device=device)  # [T]
        # map to 1..12
        month = torch.clamp((t_rel * 12).floor().long() + 1, 1, 12)  # [T]
        return month.unsqueeze(0).expand(B, -1)  # [B, T]

    def forward(self, x: torch.Tensor, month_idx: torch.Tensor | None = None):
        B, V, T, H, W = x.shape
        device = x.device
        if month_idx is None:
            month_idx = self.month_from_relative(B, T, device)   # [B, T]

        # month_idx==0 表示“禁用时间嵌入”（用于 CFG dropout/无条件分支）
        month_idx = month_idx.to(device=device, dtype=torch.long)
        mask = (month_idx > 0).to(x.dtype)                       # [B, T]

        # 将 1..12 -> 0..11，0 仍保持 0（随后与 mask 相乘即可消除影响）
        m0 = month_idx.clamp(min=1, max=12) - 1                  # [B, T]

        # lookup -> [B, T, V]
        bias_bt_v = self.weight(m0)                              # [B, T, V]

        # month_idx 通常来自数据集的元信息，其中每个时间步都给定了真实月份
        # （取值 1-12）。若数据集中缺失该字段，则会 fallback 到 month_from_relative
        # 根据窗口内相对位置近似月份。

        # 对 mask=0 的样本直接清零，防止引入季节性偏置
        bias_bt_v = bias_bt_v * mask.unsqueeze(-1)

        # reshape -> [B, V, T, 1, 1] -> broadcast 到 H,W
        bias = bias_bt_v.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B,V,T,1,1]
        return x + bias
