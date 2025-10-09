import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F
# from diffusion_icenet import IceNetUNet
# from diffusion_icenet import IceNetDataset  # 你前面定义的数据加载器
from uvit import UViT
from dataloader import ClimateForecastDataset
from SICForecast import SeaIceForecastPipeline
from SICForecast import NoiseScheduleVP
from torch.utils.data import Subset

torch.manual_seed(42)
def histogram_kl_loss(pred, target, bins=20, eps=1e-8):
    """
    计算预测结果与目标图像的像素值分布之间的 KL 散度。
    """
    B = pred.shape[0]  # batch size
    loss = 0.0
    for i in range(B):
        pred_hist = torch.histc(pred[i], bins=bins, min=0.0, max=1.0)
        target_hist = torch.histc(target[i], bins=bins, min=0.0, max=1.0)

        # 归一化成概率分布
        pred_prob = pred_hist / (pred_hist.sum() + eps)
        target_prob = target_hist / (target_hist.sum() + eps)

        # 避免 log(0)
        pred_prob = torch.clamp(pred_prob, min=eps)
        target_prob = torch.clamp(target_prob, min=eps)

        kl_div = F.kl_div(pred_prob.log(), target_prob, reduction='sum')  # KL(p || q)
        loss += kl_div

    return loss / B



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
schedule = NoiseScheduleVP(schedule='cosine')


def q_sample(x_start, t, noise, schedule):
    log_alpha = schedule.marginal_log_mean_coeff(t)
    sigma = schedule.marginal_std(t)
    return torch.exp(log_alpha)[..., None, None, None] * x_start + sigma[..., None, None, None] * noise


# === 配置参数 ===
#
# BATCH_SIZE = 2
EPOCHS_PRETRAIN = 10
EPOCHS_FINETUNE = 300
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = './icenet_diffusion.pt'

# === 加载模型 ===
# model = IceNetUNet(in_channels=50, out_months=6).to(DEVICE)
model = UViT(img_size=432, patch_size=16, in_chans=30).to(DEVICE)
pipeline = SeaIceForecastPipeline(model=model, device='cuda')


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === 加载数据 ===
root_dir = '/data/diffusionDemo/dataset1'
variables = ['psl/anom', 'siconca/abs', 'tas/anom' , 'tos/anom']  # 替换为所有所需变量名，共12个
target_var = 'siconca/abs'
reanal_dataset = ClimateForecastDataset(
        root_dir=root_dir,
        variables=variables,
        target_var=target_var,
        input_seq_len=6,
        output_seq_len=6,
        mode='obs'
    )
num_exclude = 120
total_len = len(reanal_dataset)
keep_indices = list(range(total_len - num_exclude))
reanal_dataset_trimmed = Subset(reanal_dataset, keep_indices)

reanal_dataloader = DataLoader(reanal_dataset_trimmed, batch_size=16, shuffle=True, num_workers=0)

# === 预训练 on CMIP6 ===
transfer_learning = True  
model_name = [
    'EC-Earth3/r2i1p1f1',
    'EC-Earth3/r7i1p1f1',
    'EC-Earth3/r10i1p1f1',
    'EC-Earth3/r12i1p1f1',
    'EC-Earth3/r14i1p1f1',
    'MRI-ESM2-0/r1i1p1f1',
    'MRI-ESM2-0/r2i1p1f1',
    'MRI-ESM2-0/r3i1p1f1',
    'MRI-ESM2-0/r4i1p1f1',
    'MRI-ESM2-0/r5i1p1f1'
]

cmip_dataset = ClimateForecastDataset(
        root_dir=root_dir,
        variables=variables,
        target_var=target_var,
        input_seq_len=6,
        output_seq_len=6,
        mode='transfer',
        model_names=model_name
    )
    
    
cmip_dataloader = DataLoader(cmip_dataset, batch_size=16, shuffle=True, num_workers=0)





if transfer_learning == True:
    print("\n[Stage 1] Pretraining on CMIP6 data...")
    for epoch in range(EPOCHS_PRETRAIN):
        model.train()
        running_loss = 0
        for x, y in tqdm(cmip_dataloader):
            x, y = x.to(device), y.to(device)  # [B, V, T_in, H, W], [B, T_out, H, W]
            B, V, T_in, H, W = x.shape
            T_out = y.shape[1]
    
            # 拼接条件图像
            x_cond = x.permute(0, 2, 1, 3, 4).contiguous().view(B, V * T_in, H, W)
    
            # 添加噪声
            x_start = y
            t = torch.rand(B, device=device) * schedule.T
            noise = torch.randn_like(x_start)
            x_t = q_sample(x_start, t, noise, schedule)
    
            # 拼接输入
            x_full = torch.cat([x_cond, x_t], dim=1)  # [B, 30, H, W]
            pred_noise = model(x_full, t)
    
            # 损失计算
            log_alpha = schedule.marginal_log_mean_coeff(t)
            sigma = schedule.marginal_std(t)
            x_pred = (x_t - sigma[..., None, None, None] * pred_noise) / torch.exp(log_alpha)[..., None, None, None]
            
            # 损失1：标准MSE预测噪声
            loss_denoise = F.mse_loss(pred_noise, noise)
            
            # 损失2：直方图匹配损失（在预测图与 ground-truth 图之间）
            loss_hist = histogram_kl_loss(x_pred, x_start)
            
            # 总损失（调整权重系数）
            lambda_hist = 0.2
            loss = loss_denoise + lambda_hist * loss_hist
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS_PRETRAIN} - Loss: {running_loss / len(cmip_dataloader):.4f}")
    
    torch.save(model.state_dict(), SAVE_PATH.replace('.pt', '_pretrained.pt'))

# === 微调 on 观测数据 ===
print("\n[Stage 2] Fine-tuning on observational data...")
for param in model.parameters():
    param.requires_grad = True  # 解冻所有参数


for epoch in range(EPOCHS_FINETUNE):
    model.train()
    running_loss = 0

    for x, y in tqdm(reanal_dataloader):  # x: [B, V, T, H, W], y: [B, 6, H, W]
        x, y = x.to(device), y.to(device)  # [B, V, T_in, H, W], [B, T_out, H, W]
        B, V, T_in, H, W = x.shape
        T_out = y.shape[1]

        # 拼接条件图像
        x_cond = x.permute(0, 2, 1, 3, 4).contiguous().view(B, V * T_in, H, W)

        # 添加噪声
        x_start = y
        t = torch.rand(B, device=device) * schedule.T
        noise = torch.randn_like(x_start)
        x_t = q_sample(x_start, t, noise, schedule)

        # 拼接输入
        x_full = torch.cat([x_cond, x_t], dim=1)  # [B, 30, H, W]
        pred_noise = model(x_full, t)

        # 损失计算
        log_alpha = schedule.marginal_log_mean_coeff(t)
        sigma = schedule.marginal_std(t)
        x_pred = (x_t - sigma[..., None, None, None] * pred_noise) / torch.exp(log_alpha)[..., None, None, None]
        lambda_hist = 0.2
        loss_denoise = F.mse_loss(pred_noise, noise)
        loss_hist = histogram_kl_loss(x_pred, x_start)
        loss = loss_denoise + lambda_hist * loss_hist

        running_loss += loss.item()

    avg_loss = running_loss / len(reanal_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS_FINETUNE} - Loss: {avg_loss:.6f}")

# 保存最终模型
torch.save(model.state_dict(), SAVE_PATH)
print("\n? Training complete. Model saved to", SAVE_PATH)
