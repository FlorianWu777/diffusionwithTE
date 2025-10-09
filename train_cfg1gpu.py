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
    for i in range(min(4, B)):
        nrows = 4 if x_t is not None else 3
        fig, axes = plt.subplots(nrows, T, figsize=(3 * T, 3 * nrows))
        for t in range(T):
            axes[0, t].imshow(y_true[i, t].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[0, t].set_title(f"True t={t}")
            axes[0, t].axis('off')

            axes[1, t].imshow(y_pred[i, t].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[1, t].set_title(f"Pred t={t}")
            axes[1, t].axis('off')

            axes[2, t].imshow(y_true[i, t]-y_pred[i,t].cpu(), cmap='viridis', vmin=-1, vmax=1)
            axes[2, t].set_title(f"Dif={t}")
            axes[2, t].axis('off')

            if x_t is not None:
                axes[3, t].imshow(x_t[i, t].cpu(), cmap='viridis', vmin=0, vmax=1)
                axes[3, t].set_title(f"Noised t={t}")
                axes[3, t].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stage}_epoch{epoch+1}_sample{i}.png"))
        plt.close()


def guided_sample(unet, x_t, t, cond_feat, guidance_scale=3.0, use_cfg=True):
    if not use_cfg:
        return unet(x_t, t=t, cond=cond_feat)
    pred_cond = unet(x_t, t=t, cond=cond_feat)
    pred_uncond = unet(x_t, t=t, cond=torch.zeros_like(cond_feat))
    return (1 + guidance_scale) * pred_cond - guidance_scale * pred_uncond


def evaluate_on_validation(valid_loader, encoder, unet, schedule, device):
    encoder.eval()
    unet.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            B, V, T_in, H, W = x.shape
            cond_feat = encoder(x)
            noise = torch.randn_like(y)
            t = torch.rand(B, device=device) * schedule.T
            x_t, alpha, sigma = q_sample(y, t, noise, schedule)
            pred_noise = guided_sample(unet, x_t, t, cond_feat)
            loss_denoise = F.mse_loss(pred_noise, noise)
            x0_pred = (x_t - sigma[..., None, None, None] * pred_noise) / alpha[..., None, None, None]
            loss_hist = histogram_kl_loss(x0_pred, y)
            total_loss += (loss_denoise + 0.1 * loss_hist).item()
    return total_loss / len(valid_loader)


def train_loop(dataloader, valid_loader, encoder, unet, optimizer, schedule, device, writer, epochs, desc, save_path):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        encoder.train()
        unet.train()
        for x, y in tqdm(dataloader, desc=f"[{desc}] Epoch {epoch+1}"):
            print(f"[Rank {dist.get_rank()}] :x={x.shape}, y={y.shape}")
            x, y = x.to(device), y.to(device)
            B, V, T_in, H, W = x.shape
            cond_feat = encoder(x)
            if torch.rand(1).item() < 0.1:
                cond_feat = torch.zeros_like(cond_feat)
            noise = torch.randn_like(y)
            t_max = schedule.T * min(1.0, (epoch + 1) / epochs)
            t = torch.rand(B, device=device) * t_max
            x_t, alpha, sigma = q_sample(y, t, noise, schedule)
            pred_noise = unet(x_t, t=t, cond=cond_feat)
            loss_denoise = F.mse_loss(pred_noise, noise)
            x0_pred = (x_t - sigma[..., None, None, None] * pred_noise) / alpha[..., None, None, None]
            loss_hist = histogram_kl_loss(x0_pred, y)
            loss = loss_denoise + 0.1 * loss_hist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar(f"{desc}/MSE", loss_denoise.item(), epoch)
        writer.add_scalar(f"{desc}/KL", loss_hist.item(), epoch)
        plot_predictions(y, x0_pred, x_t, epoch, desc)
        print(f"[{desc}] Epoch {epoch+1}: MSE={loss_denoise.item():.4f}, KL={loss_hist.item():.4f}")

        val_loss = evaluate_on_validation(valid_loader, encoder, unet, schedule, device)
        writer.add_scalar(f"{desc}/Val_Loss", val_loss, epoch)
        print(f"[{desc}] Epoch {epoch+1}: Validation Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'unet': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss
            }, save_path)
            print(f"? Best model saved at epoch {epoch+1} with Val_Loss={val_loss:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())

    schedule = NoiseScheduleVP(schedule='cosine')
    writer = SummaryWriter(log_dir="./runs/conditional_diffusion")

    swin_config = {
        'in_chans': 4,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (1, 12, 12),
        'patch_size': (1, 6, 6),
        'patch_norm': True,
        'pretrained': None
    }

    encoder = SwinConditionEncoder(swin_config, output_dim=256).to(device)
    unet = CrossAttentionUNet(in_channels=1, cond_dim=256).to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(unet.parameters()), lr=1e-4)

    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = 'siconca/abs'
    cmip_names = ['EC-Earth3/r2i1p1f1', 'MRI-ESM2-0/r1i1p1f1']

    cmip_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names)
    cmip_loader = DataLoader(cmip_dataset, batch_size=4, shuffle=True, num_workers=4)

    reanal_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'obs')
    total_len = len(reanal_dataset)
    train_indices = list(range(0, total_len - 156))
    valid_indices = list(range(total_len - 156, total_len - 120))
    test_indices = list(range(total_len - 120, total_len))

    reanal_train_set = Subset(reanal_dataset, train_indices)
    reanal_valid_set = Subset(reanal_dataset, valid_indices)
    reanal_test_set = Subset(reanal_dataset, test_indices)

    reanal_train_loader = DataLoader(reanal_train_set, batch_size=4, shuffle=True, num_workers=4)
    reanal_valid_loader = DataLoader(reanal_valid_set, batch_size=4, shuffle=False, num_workers=4)
    reanal_test_loader = DataLoader(reanal_test_set, batch_size=4, shuffle=False, num_workers=4)

    train_loop(cmip_loader, reanal_valid_loader, encoder, unet, optimizer, schedule, device, writer, epochs=100, desc="Pretrain", save_path='./model_pretrain.pth')
    torch.save({'encoder': encoder.state_dict(), 'unet': unet.state_dict()}, 'swin_unet_pretrained.pt')

    train_loop(reanal_train_loader, reanal_valid_loader, encoder, unet, optimizer, schedule, device, writer, epochs=300, desc="Finetune", save_path='./model_finetune.pth')
    torch.save({'encoder': encoder.state_dict(), 'unet': unet.state_dict()}, 'swin_unet.pt')

    writer.close()


if __name__ == '__main__':
    main()
