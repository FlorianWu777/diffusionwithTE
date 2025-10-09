import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from uvit import UViT
from dataloader import ClimateForecastDataset
from SICForecast import SeaIceForecastPipeline
from SICForecast import NoiseScheduleVP

# 设置设备与调度器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
schedule = NoiseScheduleVP(schedule='cosine')

# DDIM-style 采样函数（从噪声生成 x0）
def ddim_sample(model, x_cond, schedule, shape, device, steps=50, eta=0.1):
    B, T_out, H, W = shape
    x = torch.randn(B, T_out, H, W, device=device)
    ts = torch.linspace(0.8, 1e-3, steps, device=device)

    for t in ts:
        t_tensor = torch.full((B,), t, device=device)
        log_alpha = schedule.marginal_log_mean_coeff(t_tensor)
        sigma = schedule.marginal_std(t_tensor)

        x_full = torch.cat([x_cond, x], dim=1)
        pred_noise = model(x_full, t_tensor)

        x_0_pred = (x - sigma[:, None, None, None] * pred_noise) / torch.exp(log_alpha[:, None, None, None])

        noise = torch.randn_like(x)
        x = (
            torch.exp(log_alpha[:, None, None, None]) * x_0_pred +
            eta * sigma[:, None, None, None] * noise
        )

    return x_0_pred

# 数据集加载
root_dir = '/data/diffusionDemo/dataset1'
variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']  # 替换为你使用的所有变量
target_var = 'siconca/abs'

val_dataset = ClimateForecastDataset(
    root_dir=root_dir,
    variables=variables,
    target_var=target_var,
    input_seq_len=6,
    output_seq_len=6,
    mode='obs'
)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# 加载模型
model_path = "/data/diffusionDemo/icenet_diffusion.pt"  # 选一个模型即可
model = UViT(img_size=432, patch_size=16, in_chans=30).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 推理并保存 ensemble 输出
all_ensemble_preds = []
all_first_members = []

ensemble_size = 25
output_dir = './ensemble_outputs'
os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for x, y in tqdm(val_dataloader, desc="Conditional DDIM Ensemble Sampling"):
        x = x.to(device)  # [B, V, T_in, H, W]
        B, V, T_in, H, W = x.shape
        T_out = y.shape[1]

        # 构造条件输入 [B, V*T_in, H, W] → [B, 24, H, W] (假设 4 变量 * 6 时刻)
        x_cond = x.permute(0, 2, 1, 3, 4).contiguous().view(B, V * T_in, H, W)

        member_preds = []
        for _ in range(ensemble_size):
            x0_pred = ddim_sample(model, x_cond, schedule, shape=y.shape, device=device)
            member_preds.append(x0_pred.cpu())

        # 转为 [B, 25, 6, H, W]
        member_preds = torch.stack(member_preds, dim=0).permute(1, 0, 2, 3, 4)
        all_ensemble_preds.append(member_preds)

        # 提取第一个成员的第6步（索引5）[B, H, W]
        first_member = member_preds[:, 0, 5, :, :]
        all_first_members.append(first_member)

# 拼接所有 batch
final_output = torch.cat(all_ensemble_preds, dim=0).numpy()       # [N, 25, 6, H, W]
first_members_output = torch.cat(all_first_members, dim=0).numpy()  # [N, H, W]

# 保存结果
np.save('./conditional_ensemble_members.npy', final_output)
np.save('./first_ensemble_member.npy', first_members_output)

print("✅ 已保存所有 ensemble 成员，文件: conditional_ensemble_members.npy")
print("✅ 已保存第一个 ensemble 成员的第6步，文件: first_ensemble_member.npy")
