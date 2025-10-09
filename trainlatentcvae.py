import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import ClimateForecastDataset
from swindiffusionmodel import SwinConditionEncoder
from caunet import CrossAttentionUNet
from SICForecast import NoiseScheduleVP
from piqa import SSIM
from residualDecoder import CNNDecoder


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.autograd.set_detect_anomaly(True)

class PosteriorNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, z_y):
        mu = self.fc_mu(z_y)
        logvar = self.fc_logvar(z_y)
        return mu, logvar

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def histogram_kl_loss(pred, target, bins=20, eps=1e-8):
    B = pred.shape[0]
    loss = 0.0
    for i in range(B):
        vmin = torch.min(pred[i]).item()
        vmax = torch.max(pred[i]).item()
        if vmin == vmax:
            vmax += 1e-5  # 防止 histc 崩溃
        pred_hist = torch.histc(pred[i], bins=bins, min=vmin, max=vmax)
        target_hist = torch.histc(target[i], bins=bins, min=vmin, max=vmax)
        pred_prob = pred_hist / (pred_hist.sum() + eps)
        target_prob = target_hist / (target_hist.sum() + eps)
        pred_prob = torch.clamp(pred_prob, min=eps)
        target_prob = torch.clamp(target_prob, min=eps)
        kl_div = F.kl_div(pred_prob.log(), target_prob, reduction='sum')
        loss += kl_div
    return loss / B



def plot_predictions(y_true, y_pred, epoch, stage, save_dir="./plotslossmod"):
    if dist.get_rank() != 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    B,C, H, W = y_true.shape
    print(y_true.shape)
    for i in range(min(4, B)):
        fig, axes = plt.subplots(3, C, figsize=(3 * C, 9))
        axes = axes.reshape(3, C)

        for t in range(C):
            axes[0, t].imshow(y_true[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[0, t].set_title(f"True t={t}")
            axes[0, t].axis('off')

            axes[1, t].imshow(y_pred[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[1, t].set_title(f"Pred t={t}")
            axes[1, t].axis('off')

            diff = y_true[i, t].detach().cpu() - y_pred[i, t].detach().cpu()
            axes[2, t].imshow(diff, cmap='viridis', vmin=-1, vmax=1)
            axes[2, t].set_title(f"Diff t={t}")
            axes[2, t].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stage}_epoch{epoch+1}_sample{i}.png"))
        plt.close()


def evaluate_on_validation(valid_loader, encoder_x, decoder, device):
    encoder_x.eval()
    decoder.eval()

    total_loss = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            B, V, T_in, H, W = x.shape
  
            z_x = encoder_x(x)  # [B, D]
            y_pred = decoder(z_x)  # [B, T_out, H, W] or similar

            loss_recon = F.mse_loss(y_pred.squeeze(), y.squeeze())
            total_loss += loss_recon.item()

    return total_loss / len(valid_loader)





# ========== ViT-based 双编码器训练流程 ==========
def train_loop(dataloader, valid_loader, encoder_x, encoder_y, posterior_net, decoder, optimizer, device, writer, epochs, desc, save_path):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        dataloader.sampler.set_epoch(epoch)

        encoder_x.train()
        encoder_y.train()
        posterior_net.train()
        decoder.train()

        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"[{desc}] Epoch {epoch+1}", disable=dist.get_rank() != 0)):
            try:
                x, y = x.to(device), y.to(device)
                # 编码阶段
                z_x = encoder_x(x)
                z_y = encoder_y(y)
                mu, logvar = posterior_net(z_y)
                z = reparameterize(mu, logvar)

                #latent = torch.concat([z_x, z] ,dim=-2)
                latent = z_x + z

                # 解码阶段
                y_pred = decoder(latent)

                # Loss 计算
                loss_recon = F.mse_loss(y_pred.squeeze(), y.squeeze())
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss_recon + 1e-3 * kl_loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                dist.barrier()

            except Exception as e:
                print(f"[RANK {dist.get_rank()}] ? Exception during training: {e}")
                import traceback
                traceback.print_exc()
                raise e

        if dist.get_rank() == 0:
            writer.add_scalar(f"{desc}/Recon_Loss", loss_recon.item(), epoch)
            writer.add_scalar(f"{desc}/KL_Loss", kl_loss.item(), epoch)
            print(f"[{desc}] Epoch {epoch+1}: Recon={loss_recon.item():.4f}, KL={kl_loss.item():.4f}")

            # 验证阶段
            val_loss = evaluate_on_validation(valid_loader, encoder_x, decoder, device)
            writer.add_scalar(f"{desc}/Val_Loss", val_loss, epoch)
            plot_predictions(y.squeeze(), y_pred.squeeze(), epoch, desc)
            print(f"[{desc}] Epoch {epoch+1}: Validation Loss = {val_loss:.4f}")

            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'encoder_x': encoder_x.state_dict(),
                    'encoder_y': encoder_y.state_dict(),
                    'posterior_net': posterior_net.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss
                }, save_path)
                print(f"? Best model saved at epoch {epoch+1} with Val_Loss={val_loss:.4f}")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    writer = SummaryWriter(log_dir="./runs/conditional_cvae") if local_rank == 0 else None
    print(torch.cuda.get_device_name())

    latent_dim = 256
    swin_config = {
        'in_chans': 4,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': (1, 4, 4),
        'patch_size': (1, 4, 4),
        'patch_norm': True,
        'pretrained': None
    }

    encoder_x = SwinConditionEncoder(swin_config, output_dim=latent_dim, pool='mean').to(device)
    encoder_y = SwinConditionEncoder(swin_config, output_dim=latent_dim, pool='mean').to(device)
    decoder = CNNDecoder(input_dim=latent_dim, out_channels=4).to(device)
    posterior_net = PosteriorNet(latent_dim).to(device)

    encoder_x = DDP(encoder_x, device_ids=[local_rank])
    encoder_y = DDP(encoder_y, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])
    posterior_net = DDP(posterior_net, device_ids=[local_rank])

    optimizer = optim.Adam(list(encoder_x.parameters()) + list(encoder_y.parameters()) + list(posterior_net.parameters()) + list(decoder.parameters()), lr=1e-4)

    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']

    cmip_names = ['EC-Earth3/r2i1p1f1', 'MRI-ESM2-0/r1i1p1f1']
    cmip_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names)
    cmip_sampler = DistributedSampler(cmip_dataset, shuffle=True, drop_last=True)
    cmip_loader = DataLoader(cmip_dataset, batch_size=4, sampler=cmip_sampler, num_workers=4)

    reanal_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'obs')
    total_len = len(reanal_dataset)

    from torch.utils.data import Subset
    train_indices = list(range(0, total_len - 156))
    valid_indices = list(range(total_len - 156, total_len - 120))
    test_indices = list(range(total_len - 120, total_len))

    reanal_train_set = Subset(reanal_dataset, train_indices)
    reanal_valid_set = Subset(reanal_dataset, valid_indices)
    reanal_test_set = Subset(reanal_dataset, test_indices)

    reanal_train_sampler = DistributedSampler(reanal_train_set)
    reanal_valid_sampler = DistributedSampler(reanal_valid_set)
    reanal_test_sampler = DistributedSampler(reanal_test_set)

    reanal_train_loader = DataLoader(reanal_train_set, batch_size=2, sampler=reanal_train_sampler, num_workers=0)
    reanal_valid_loader = DataLoader(reanal_valid_set, batch_size=2, sampler=reanal_valid_sampler, num_workers=0)
    reanal_test_loader = DataLoader(reanal_test_set, batch_size=2, sampler=reanal_test_sampler, num_workers=0)

    torch.autograd.set_detect_anomaly(True)

    train_loop(cmip_loader, reanal_valid_loader, encoder_x, encoder_y, posterior_net, decoder, optimizer, device, writer, epochs=50, desc="Pretrain", save_path='./model/pretrain.pt')

    if local_rank == 0:
        torch.save({'encoder_x': encoder_x.module.state_dict(), 'encoder_y': encoder_y.module.state_dict(), 'posterior_net': posterior_net.module.state_dict(), 'decoder': decoder.module.state_dict()}, 'swin_cvae_pretrained.pt')

    train_loop(reanal_train_loader, reanal_valid_loader, encoder_x, encoder_y, posterior_net, decoder, optimizer, device, writer, epochs=100, desc="Finetune", save_path='./model/finetune_best.pt')

    if local_rank == 0:
        torch.save({'encoder_x': encoder_x.module.state_dict(), 'encoder_y': encoder_y.module.state_dict(), 'posterior_net': posterior_net.module.state_dict(), 'decoder': decoder.module.state_dict()}, 'swin_cvae.pt')

    if writer:
        writer.close()
    dist.destroy_process_group()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print("error that:", e)
        traceback.print_exc()

