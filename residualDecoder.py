import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.block(x) + x)

class CNNDecoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, out_channels=1):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim * 27 * 27)
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 27 ¡ú 54
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),  # 54 ¡ú 108
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),  # 108 ¡ú 216
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2),  # 216 ¡ú 432
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU()
        )
        self.out_layer = nn.Conv2d(64, out_channels, 1)

    def forward(self, z):
        B = z.size(0)
        x = self.project(z).view(B, 128, 27, 27)
        x = self.blocks(x)
        x = self.upsample(x)
        return self.out_layer(x)  # shape: [B, out_channels, 432, 432]
