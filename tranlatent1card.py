import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import ClimateForecastDataset
from swindiffusionmodel import SwinConditionEncoder
from caunet import CrossAttentionUNet
from SICForecast import NoiseScheduleVP

def q_sample(x_start, t, noise, schedule):
    alpha = schedule.marginal_mean_coeff(t)
    sigma = schedule.marginal_std(t)
    return alpha[..., None, None, None] * x_start + sigma[..., None, None, None] * noise, alpha, sigma


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


def plot_predictions(y_true, y_pred, x_t, epoch, stage, save_dir="./plotslossmod"):
    os.makedirs(save_dir, exist_ok=True)
    B, T, H, W = y_true.shape
    for i in range(min(3, B)):
        nrows = 3 if x_t is not None else 2
        fig, axes = plt.subplots(nrows, max(T,2), figsize=(3 * max(T,2), 3 * nrows))
        for t in range(T):
            
            axes[0, t].imshow(y_true[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[0, t].set_title(f"True t={t}")
            axes[0, t].axis('off')

            axes[1, t].imshow(y_pred[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[1, t].set_title(f"Pred t={t}")
            axes[1, t].axis('off')

            if x_t is not None:
                axes[2, t].imshow(x_t[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
                axes[2, t].set_title(f"Noised t={t}")
                axes[2, t].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stage}_epoch{epoch+1}_sample{i}.png"))
        plt.close()


def train_loop(dataloader, encoder, unet, optimizer, schedule, device, writer, epochs, desc):
    for epoch in range(epochs):
        encoder.train()
        unet.train()
        for x, y in tqdm(dataloader, desc=f"[{desc}] Epoch {epoch+1}"):

            x, y = x.to(device), y.to(device)
            B, V, T_in, H, W = x.shape
            T_out = y.shape[1]

            cond_feat = encoder(x)
            noise = torch.randn_like(y)
            t_max = schedule.T * min(1.0, (epoch + 1) / epochs)
            t = torch.rand(B, device=device) * t_max
            x_t, alpha, sigma = q_sample(y, t, noise, schedule)

            pred = unet(x_t, t=t, cond=cond_feat)
            pred_noise = pred

            loss_denoise = F.mse_loss(pred_noise, noise)
            x0_pred = (x_t - sigma[..., None, None, None] * pred_noise) / alpha[..., None, None, None]
            loss_hist = histogram_kl_loss(x0_pred, y)

            loss = loss_denoise + 0.1 * loss_hist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar(f"{desc}/MSE", loss_denoise.item(), epoch)
        writer.add_scalar(f"{desc}/KL", loss_hist.item(), epoch)
        print()
        plot_predictions(y, x0_pred, x_t, epoch, desc)
        print(f"[{desc}] Epoch {epoch+1}: MSE={loss_denoise.item():.4f}, KL={loss_hist.item():.4f}")


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    writer = SummaryWriter(log_dir="./runs/conditional_diffusion")
    schedule = NoiseScheduleVP(schedule='linear')

    # Build model
    swin_config = {
        'in_chans': 4,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (1, 7, 7),
        'patch_size': (1, 4, 4),
        'patch_norm': True,
        'pretrained': None
    }
    encoder = SwinConditionEncoder(swin_config, output_dim=256).to(device)
    unet = CrossAttentionUNet(in_channels=1, cond_dim=256).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(unet.parameters()), lr=1e-4)

    # Dataset
    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = 'siconca/abs'

    cmip_names = ['EC-Earth3/r2i1p1f1', 'MRI-ESM2-0/r1i1p1f1']
    cmip_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names)
    cmip_loader = DataLoader(cmip_dataset, batch_size=4, shuffle=True, num_workers=4)

    reanal_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'obs')
    keep_indices = list(range(len(reanal_dataset) - 120))
    reanal_subset = Subset(reanal_dataset, keep_indices)
    reanal_loader = DataLoader(reanal_subset, batch_size=4, shuffle=True, num_workers=2)

    # Training
    train_loop(cmip_loader, encoder, unet, optimizer, schedule, device, writer, epochs=100, desc="Pretrain")
    torch.save({'encoder': encoder.state_dict(), 'unet': unet.state_dict()}, 'swin_unet_pretrained.pt')

    train_loop(reanal_loader, encoder, unet, optimizer, schedule, device, writer, epochs=300, desc="Finetune")
    torch.save({'encoder': encoder.state_dict(), 'unet': unet.state_dict()}, 'swin_unet.pt')

    writer.close()
    print("Training finished and models saved.")


if __name__ == '__main__':
    main()
