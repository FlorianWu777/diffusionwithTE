import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from uvit import UViT
from dataloader import ClimateForecastDataset
from SICForecast import SeaIceForecastPipeline
from SICForecast import NoiseScheduleVP
from torch.utils.data import Subset

# === 分布式训练设置 ===
local_rank = int(os.environ['LOCAL_RANK'])
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


torch.manual_seed(42)
def histogram_kl_loss(pred, target, bins=20, eps=1e-8):
    B = pred.shape[0]
    loss = 0.0
    for i in range(B):
        pred_hist = torch.histc(pred[i], bins=bins, min=0.0, max=1.0)
        target_hist = torch.histc(target[i], bins=bins, min=0.0, max=1.0)
        pred_prob = pred_hist / (pred_hist.sum() + eps)
        target_prob = target_hist / (target_hist.sum() + eps)
        pred_prob = torch.clamp(pred_prob, min=eps)
        target_prob = torch.clamp(target_prob, min=eps)
        kl_div = F.kl_div(pred_prob.log(), target_prob, reduction='sum')
        loss += kl_div
    return loss / B

schedule = NoiseScheduleVP(schedule='cosine')

def q_sample(x_start, t, noise, schedule):
    log_alpha = schedule.marginal_log_mean_coeff(t)
    sigma = schedule.marginal_std(t)
    return torch.exp(log_alpha)[..., None, None, None] * x_start + sigma[..., None, None, None] * noise

# === 配置参数 ===
BATCH_SIZE = 2
EPOCHS_PRETRAIN = 10
EPOCHS_FINETUNE = 300
LR = 1e-4
SAVE_PATH = './icenet_diffusion.pt'

# === 加载模型 ===
model = UViT(img_size=432, patch_size=16, in_chans=30).to(device)
model = DDP(model, device_ids=[local_rank])
pipeline = SeaIceForecastPipeline(model=model, device=device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === 加载数据 ===
root_dir = '/data/diffusionDemo/dataset1'
variables = ['psl/anom', 'siconca/abs', 'tas/anom' , 'tos/anom']
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
reanal_sampler = DistributedSampler(reanal_dataset_trimmed)
reanal_dataloader = DataLoader(reanal_dataset_trimmed, batch_size=16, sampler=reanal_sampler, num_workers=0)

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
cmip_sampler = DistributedSampler(cmip_dataset)
cmip_dataloader = DataLoader(cmip_dataset, batch_size=4, sampler=cmip_sampler, num_workers=0)

if transfer_learning:
    print("\n[Stage 1] Pretraining on CMIP6 data...")
    for epoch in range(EPOCHS_PRETRAIN):
        model.train()
        cmip_sampler.set_epoch(epoch)
        running_loss = 0

        for x, y in tqdm(cmip_dataloader, disable=local_rank != 0):
            x, y = x.to(device), y.to(device)
            B, V, T_in, H, W = x.shape
            x_cond = x.permute(0, 2, 1, 3, 4).contiguous().view(B, V * T_in, H, W)
            x_start = y
            t = torch.rand(B, device=device) * schedule.T
            noise = torch.randn_like(x_start)
            x_t = q_sample(x_start, t, noise, schedule)
            x_full = torch.cat([x_cond, x_t], dim=1)
            pred_noise = model(x_full, t)
            log_alpha = schedule.marginal_log_mean_coeff(t)
            sigma = schedule.marginal_std(t)
            x_pred = (x_t - sigma[..., None, None, None] * pred_noise) / torch.exp(log_alpha)[..., None, None, None]
            loss_denoise = F.mse_loss(pred_noise, noise)
            loss_hist = histogram_kl_loss(x_pred, x_start)
            loss = loss_denoise + 0.1 * loss_hist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if local_rank == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_PRETRAIN} - Loss: {running_loss / len(cmip_dataloader):.4f}")

    if local_rank == 0:
        torch.save(model.module.state_dict(), SAVE_PATH.replace('.pt', '_pretrained.pt'))

# === 微调 on 观测数据 ===
print("\n[Stage 2] Fine-tuning on observational data...")
for param in model.parameters():
    param.requires_grad = True

for epoch in range(EPOCHS_FINETUNE):
    model.train()
    reanal_sampler.set_epoch(epoch)
    running_loss = 0

    for x, y in tqdm(reanal_dataloader, disable=local_rank != 0):
        x, y = x.to(device), y.to(device)
        B, V, T_in, H, W = x.shape
        T_out = y.shape[1]

        x_cond = x.permute(0, 2, 1, 3, 4).contiguous().view(B, V * T_in, H, W)
        x_start = y
        t = torch.rand(B, device=device) * schedule.T
        noise = torch.randn_like(x_start)
        x_t = q_sample(x_start, t, noise, schedule)

        x_full = torch.cat([x_cond, x_t], dim=1)
        pred_noise = model(x_full, t)

        log_alpha = schedule.marginal_log_mean_coeff(t)
        sigma = schedule.marginal_std(t)
        x_pred = (x_t - sigma[..., None, None, None] * pred_noise) / torch.exp(log_alpha)[..., None, None, None]

        loss_denoise = F.mse_loss(pred_noise, noise)
        loss_hist = histogram_kl_loss(x_pred, x_start)
        loss = loss_denoise + 0.1 * loss_hist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(reanal_dataloader)
    if local_rank == 0:
        print(f"Epoch {epoch+1}/{EPOCHS_FINETUNE} - Loss: {avg_loss:.6f}")

if local_rank == 0:
    torch.save(model.module.state_dict(), SAVE_PATH)
    print("\n? Training complete. Model saved to", SAVE_PATH)

dist.destroy_process_group()