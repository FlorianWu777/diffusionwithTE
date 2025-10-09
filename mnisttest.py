import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from uvit import UViT  # 用你的 UViT 实现
from SICForecast import NoiseScheduleVP  # 用你的调度器
import os
from tqdm import tqdm

device = torch.device("cpu")

# === noise schedule & q_sample ===
schedule = NoiseScheduleVP(schedule='cosine')

def q_sample(x_start, t, noise, schedule):
    alpha = schedule.marginal_mean_coeff(t)
    sigma = schedule.marginal_std(t)
    return alpha[:, None, None, None] * x_start + sigma[:, None, None, None] * noise

def histogram_kl_loss(pred, target, bins=20, eps=1e-8):
    B = pred.shape[0]
    loss = 0.0
    for i in range(B):
        pred_hist = torch.histc(pred[i], bins=bins, min=0.0, max=1.0)
        target_hist = torch.histc(target[i], bins=bins, min=0.0, max=1.0)
        pred_prob = pred_hist / (pred_hist.sum() + eps)
        target_prob = target_hist / (target_hist.sum() + eps)
        kl_div = F.kl_div(pred_prob.log(), target_prob, reduction='sum')
        loss += kl_div
    return loss / B

# === dataset ===
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # Converts to [0,1]
])

mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=2)

# === model ===
model = UViT(img_size=32, patch_size=4, in_chans=1, out_chans=1, embed_dim=256, depth=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter("./runs/mnist_diffusion")
EPOCHS = 50

# === training ===
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, _ in tqdm(loader, desc=f"Epoch {epoch+1}"):
        x = x.to(device)  # [B, 1, 32, 32]
        noise = torch.randn_like(x)
        t = torch.rand(x.shape[0], device=device) * schedule.T
        x_t = q_sample(x, t, noise, schedule)

        x_input = torch.cat([x_t], dim=1)  # In case you add condition later
        pred_noise = model(x_input, t)

        loss_denoise = F.mse_loss(pred_noise, noise)

        # Decode x0_pred
        alpha = schedule.marginal_mean_coeff(t)
        sigma = schedule.marginal_std(t)
        x0_pred = (x_t - sigma[:, None, None, None] * pred_noise) / alpha[:, None, None, None]
        x0_pred = torch.clamp(x0_pred, 0.0, 1.0)

        loss_hist = histogram_kl_loss(x0_pred, x)
        loss = loss_denoise + 0.3 * loss_hist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    writer.add_scalar("train/total_loss", total_loss / len(loader), epoch)
    writer.add_image("train/x0_pred", x0_pred[0], epoch)
    writer.add_image("train/x_gt", x[0], epoch)
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

writer.close()
