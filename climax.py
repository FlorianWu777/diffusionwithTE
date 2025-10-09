import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 假设 ClimateForecastDataset 已经像你提供的一样定义好了
from dataloader import ClimateForecastDataset  # 注意替换成你的文件名


# 1. 定义一个小版 ClimaX 模型（只做海冰预测）
class SimpleClimaX(nn.Module):
    def __init__(self, input_vars, input_seq_len, embed_dim=256, nhead=8, depth=6, patch_size=4, img_size=432):
        super().__init__()

        self.input_vars = input_vars
        self.input_seq_len = input_seq_len
        self.patch_size = patch_size

        # 简单地降维成 embedding
        self.embedding = nn.Conv3d(
            in_channels=len(input_vars),
            out_channels=embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        # Transformer 编码器部分
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 解码器: 还原到目标图像
        num_patches = (img_size // patch_size) ** 2
        self.decoder = nn.Linear(embed_dim, num_patches)

    def forward(self, x):
        # x shape: [B, V, T_in, H, W]
        B, V, T, H, W = x.shape

        # 合并时间和变量维度到batch
        x = x.view(B, V*T, H, W)

        # (B, V, T, H, W) -> (B, embed_dim, T, H//p, W//p)
        x = self.embedding(x)

        # 展平空间维度 -> Transformer要的是序列格式
        x = x.flatten(3).transpose(2, 3)  # (B, HW, T, embed_dim)
        x = x.flatten(1, 2)  # (B, Seq_len, embed_dim)

        # Transformer编码
        x = self.transformer(x)

        # 解码输出每个patch的值
        x = self.decoder(x)  # (B, Seq_len, patch_H * patch_W)

        # 简单地reshape回来 (B, T_out, H, W)
        patch_H = patch_W = self.patch_size
        spatial_size = int(x.shape[-1] ** 0.5) * patch_H
        x = x.view(B, T, spatial_size, spatial_size)

        return x


# 2. 数据加载
root_dir = '/data/diffusionDemo/dataset1'
variables = ['rsus/anom', 'tas/anom', 'tos/anom', 'siconca/abs']  # 可以换成更多变量

train_dataset = ClimateForecastDataset(
    root_dir=root_dir,
    variables=variables,
    target_var='siconca/abs',
    input_seq_len=12,
    output_seq_len=6,
    mode='obs'
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

# 3. 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleClimaX(
    input_vars=variables,
    input_seq_len=12,
    embed_dim=256,
    nhead=8,
    depth=6,
    patch_size=4,
    img_size=432
).to(device)

# 4. 简单训练循环
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("训练完成！")