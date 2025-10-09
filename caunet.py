import torch
import torch.nn as nn
class EfficientCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, time_embed_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )

        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fuse = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

    def forward(self, x, cond, t):
        B, C, H, W = x.shape

        # Step 1: 时间编码
        t = t.view(B, 1).float()
        t_emb = self.time_embed(t)  # [B, C]
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Step 2: cond 聚合
        cond_vec = self.cond_proj(cond.mean(dim=1))  # [B, C]
        cond_map = cond_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        # Step 3: 特征融合
        fused = torch.cat([x, cond_map + t_emb], dim=1)
        out = self.fuse(fused)

        return x + self.gamma * out  # Residual

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, cond_dim, time_embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.key_proj = nn.Linear(cond_dim, hidden_dim)
        self.value_proj = nn.Linear(cond_dim, hidden_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)

    def forward(self, x, cond, t):
        """
        x: [B, C, H, W]
        cond: [B, N, D]
        t: [B] or [B, 1]
        """
        B, C, H, W = x.shape
        x_flat = self.query_proj(x).flatten(2).transpose(1, 2)  # [B, HW, C]

        # Encode timestep
        t = t.view(B, 1).float()
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, C]

        # Time-aware conditioning
        k = self.key_proj(cond)
        v = self.value_proj(cond)
        k = torch.cat([k, t_emb], dim=1)  # [B, N+1, C]
        v = torch.cat([v, t_emb], dim=1)

        out, _ = self.attn(x_flat, k, v)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + self.out_proj(out)  # Residual


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CrossAttentionUNet(nn.Module):
    def __init__(self, in_channels=1, cond_dim=128, base_channels=64):
        super().__init__()
        # Encoder
        self.down1 = DoubleConv(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)
        self.cross_attn_bottleneck = EfficientCrossAttentionBlock(base_channels * 4, cond_dim)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.cross_attn_dec2 = EfficientCrossAttentionBlock(base_channels * 2, cond_dim)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)
        self.cross_attn_dec1 = EfficientCrossAttentionBlock(base_channels, cond_dim)

        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x, t, cond):
        """
        x: [B, 1, H, W]
        cond: [B, N, D]
        t: [B]
        """
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))

        # Bottleneck Attention
        x3 = self.cross_attn_bottleneck(x3, cond, t)

        x = self.up2(x3)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.cross_attn_dec2(x, cond, t)

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))
        x = self.cross_attn_dec1(x, cond, t)

        return self.final(x)
