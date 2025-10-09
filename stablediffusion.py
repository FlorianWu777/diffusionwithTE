import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import ClimateForecastDataset
from residualDecoder import CNNDecoder

# ===== 1. VAE Encoder =====
class VAEEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, latent_dim * 2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar


# ===== 2. 3D UNet for Diffusion =====
class DiffusionUNet3D(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(latent_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        return self.net(x)


# ===== 3. Gaussian Diffusion Process =====
class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_model, timesteps=1000):
        super().__init__()
        self.model = denoise_model
        self.timesteps = timesteps
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, timesteps))
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas = self.alphas_cumprod[t].to(x_start.device) ** 0.5
        sqrt_one_minus = (1. - self.alphas_cumprod[t].to(x_start.device)) ** 0.5
        return sqrt_alphas.view(-1, 1, 1, 1, 1) * x_start + sqrt_one_minus.view(-1, 1, 1, 1, 1) * noise, noise

    def p_losses(self, x_start, t):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted_noise = self.model(x_noisy, t)
        z_denoised = x_noisy - predicted_noise  # 简化去噪公式
        loss = F.mse_loss(predicted_noise, noise)
        return loss, x_start, x_noisy, predicted_noise, z_denoised



# ===== 4. Visualization =====
def plot_predictions(y_true, y_noisy, y_pred, epoch, save_dir="./plots"):
    if dist.get_rank() != 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    B, C, T, H, W = y_true.shape
    print(B,C,T,H,W)
    for i in range(min(2, B)):
        for v in range(C):
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(y_true[i, v, T//2].cpu(), cmap='viridis')
            axes[0].set_title("True")
            axes[1].imshow(y_noisy[i, v, T//2].cpu(), cmap='viridis')
            axes[1].set_title("Noisy")
            axes[2].imshow(y_pred[i, v, T//2].cpu(), cmap='viridis')
            axes[2].set_title("Pred")
            diff = y_true[i, v, T//2].cpu() - y_pred[i, v, T//2].cpu()
            axes[3].imshow(diff, cmap='bwr', vmin=-1, vmax=1)
            axes[3].set_title("Diff")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch{epoch+1}_sample{i}_var{v}.png"))
            plt.close()


# ===== 5. Training Loop =====
def train_diffusion():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    latent_dim = 4
    encoder = VAEEncoder(in_channels=4, latent_dim=latent_dim).to(device)
    decoder = CNNDecoder(input_dim=latent_dim, out_channels=4).to(device)
    unet = DiffusionUNet3D(latent_dim=latent_dim).to(device)
    diffusion = GaussianDiffusion(denoise_model=unet).to(device)

    encoder = DDP(encoder, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])
    unet = DDP(unet, device_ids=[local_rank])

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(unet.parameters()), lr=1e-4)

    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    cmip_names = ['EC-Earth3/r2i1p1f1', 'MRI-ESM2-0/r1i1p1f1']

    dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names)
    total_len = len(dataset)
    train_set = dataset
    
    reanal_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'obs')
    total_len = len(reanal_dataset)
    valid_indices = list(range(total_len - 156, total_len - 120))
    valid_set = Subset(reanal_dataset, valid_indices)
    
    train_sampler = DistributedSampler(train_set)
    valid_sampler = DistributedSampler(valid_set)

    train_loader = DataLoader(train_set, batch_size=32, sampler=train_sampler, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=4, sampler=valid_sampler, num_workers=4)

    for epoch in range(10):
        train_sampler.set_epoch(epoch)
        for x, y in tqdm(train_loader, desc=f"[Rank {local_rank}] Epoch {epoch+1}", disable=local_rank != 0):
            print(x.shape)
            print(y.shape)
            x = x.to(device)
            y = y.to(device)
            mu, logvar = encoder(x)
            z = mu.view(mu.size(0), latent_dim, 1, 1, 1).expand(-1, -1, 12, 32, 64)

            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
            loss, _, x_noisy, _, z_denoised = diffusion.p_losses(z, t)
            y_pred = decoder(z_denoised)
            recon_loss = F.mse_loss(y_pred, y)


            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

        if local_rank == 0:
            print(f"[Rank {local_rank}] Epoch {epoch+1} Loss: {loss.item():.4f}")
            plot_predictions(y_true.detach().cpu(), y_noisy.detach().cpu(), y_pred.detach().cpu(), epoch)

            # Evaluate on validation set
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                unet.eval()
                val_loss_total = 0
                count = 0
                for x_val, y_val in valid_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    mu_val, _ = encoder(x_val)
                    z_val = mu_val.view(mu_val.size(0), latent_dim, 1, 1, 1).expand(-1, -1, 12, 32, 64)
                    t_val = torch.randint(0, diffusion.timesteps, (x_val.size(0),), device=device).long()
                    val_loss, _, _, _, z_denoised = diffusion.p_losses(z_val, t_val)
                    y_val_pred = decoder(z_denoised)
                    recon_loss = F.mse_loss(y_val_pred, y_val)

                    val_loss_total += val_loss.item()
                    count += 1
                avg_val_loss = val_loss_total / count
                print(f"[Rank {local_rank}] Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

    dist.destroy_process_group()


if __name__ == '__main__':
    train_diffusion()
