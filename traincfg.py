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



# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# torch.autograd.set_detect_anomaly(True)


def q_sample(x_start, t, noise, schedule):
    alpha = schedule.marginal_mean_coeff(t)
    sigma = schedule.marginal_std(t)
    return alpha[..., None, None, None] * x_start + sigma[..., None, None, None] * noise, alpha, sigma


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



def plot_predictions(y_true, y_pred, x_t, epoch, stage, save_dir="./plotslossmod"):
    if dist.get_rank() != 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    B, T, H, W = y_true.shape
    for i in range(min(4, B)):
        nrows = 4 if x_t is not None else 3
        fig, axes = plt.subplots(nrows, T, figsize=(3 * T, 3 * nrows))
        axes = axes.reshape(nrows, T)

        for t in range(T):
            axes[0, t].imshow(y_true[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[0, t].set_title(f"True t={t}")
            axes[0, t].axis('off')

            axes[1, t].imshow(y_pred[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[1, t].set_title(f"Pred t={t}")
            axes[1, t].axis('off')


            axes[2, t].imshow(y_true[i, t].detach().cpu()-y_pred[i,t].detach().cpu(), cmap='viridis', vmin=-1, vmax=1)
            axes[2, t].set_title(f"Dif={t}")
            axes[2, t].axis('off')
            
            if x_t is not None:
                axes[3, t].imshow(x_t[i, t].detach().cpu(), cmap='viridis', vmin=0, vmax=1)
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
            T_out = y.shape[1]
    
            cond_feat = encoder(x)
            noise = torch.randn_like(y)
            t = torch.rand(B, device=device) * schedule.T
            x_t, alpha, sigma = q_sample(y, t, noise, schedule)
    
            pred = unet(x_t, t=t, cond=cond_feat)
            guidance_scale = 3.0  # 可调参数
            use_cfg = True
            
            pred_noise = guided_sample(unet, x_t, t, cond_feat, guidance_scale, use_cfg)
    
            loss_denoise = F.mse_loss(pred_noise, noise)
    
            x0_pred = (x_t - sigma[..., None, None, None] * pred_noise) / alpha[..., None, None, None]
            loss_hist = histogram_kl_loss(x0_pred, y)
    
            total_loss += (loss_denoise + 0.1 * loss_hist).item()

    return total_loss / len(valid_loader)


    
def train_loop(dataloader, valid_loader, encoder, unet, optimizer, schedule, device, writer, epochs, desc, save_path):
    best_val_loss = float('inf')
    ssim = SSIM(n_channels=1).to(device)
    for epoch in range(epochs):
        dataloader.sampler.set_epoch(epoch)

        encoder.train()
        unet.train()
        for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"[{desc}] Epoch {epoch+1}", disable=dist.get_rank() != 0)):
            try:
                x, y = x.to(device), y.to(device)
                B, V, T_in, H, W = x.shape
                T_out = y.shape[1]
        
                # === 编码条件 ===
                cond_feat = encoder(x)
        
                # === Dropout: 有一定概率使用全0条件（模拟 uncond 分支） ===
                drop_tensor = torch.rand(1, device=device)
                dist.broadcast(drop_tensor, src=0)  # 同步所有进程
                use_cfg = True
                if drop_tensor.item() < 0.1:
                    cond_feat = torch.zeros_like(cond_feat)
                    use_cfg = False  # 直接不使用 CFG 混合
        
                # === 添加噪声 ===
                noise = torch.randn_like(y)
                t_max = schedule.T * min(1.0, (epoch + 1) / epochs)
                t = torch.rand(B, device=device) * t_max
                x_t, alpha, sigma = q_sample(y, t, noise, schedule)
        
                # === 使用 guided_sample CFG 预测噪声 ===
                guidance_scale = 5.5
                pred_noise = guided_sample(unet, x_t, t, cond_feat, guidance_scale=guidance_scale, use_cfg=use_cfg)
        
                # === Loss 计算 ===
                loss_denoise = F.mse_loss(pred_noise, noise)
                x0_pred = (x_t - sigma[..., None, None, None] * pred_noise) / alpha[..., None, None, None]
                loss_hist = histogram_kl_loss(x0_pred, y)
                loss = loss_denoise + 0.1 * loss_hist
                x0_pred_clipped = torch.clamp(x0_pred, 0.0, 1.0)
                y_clipped = torch.clamp(y, 0.0, 1.0)
                loss_ssim = 1 - ssim(x0_pred_clipped, y_clipped)
                # loss_ssim = 1 - ssim(x0_pred, y)
                def high_freq_loss(x):
                    # 输入形状: [B, T, H, W]
                    fft = torch.fft.fft2(x, dim=(-2, -1))
                    fft = torch.fft.fftshift(fft)
                    magnitude = torch.abs(fft)
                    high_freq_energy = magnitude[..., H//4:, W//4:].mean()  # 只取右下角高频
                    return high_freq_energy
                loss_high_freq =  high_freq_loss(x0_pred)
                # loss += 0.5*loss_ssim
        
                # === 反向传播 ===
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                dist.barrier()  # 同步所有进程，防止 DDP 死锁
        
            except Exception as e:
                print(f"[RANK {dist.get_rank()}] ❌ Exception during training: {e}")
                import traceback
                traceback.print_exc()
                raise e

        if dist.get_rank() == 0:
            writer.add_scalar(f"{desc}/MSE", loss_denoise.item(), epoch)
            writer.add_scalar(f"{desc}/KL", loss_hist.item(), epoch)
            plot_predictions(y, x0_pred, x_t, epoch, desc)
            print(f"[{desc}] Epoch {epoch+1}: MSE={loss_denoise.item():.4f}, KL={loss_hist.item():.4f}, highfreq={loss_high_freq}")

            # === 验证 ===
            val_loss = evaluate_on_validation(valid_loader, encoder, unet, schedule, device)
            writer.add_scalar(f"{desc}/Val_Loss", val_loss, epoch)
            print(f"[{desc}] Epoch {epoch+1}: Validation Loss = {val_loss:.4f}")

            # === 保存最佳模型 ===
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'unet': unet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss
                }, save_path)
                print(f"✅ Best model saved at epoch {epoch+1} with Val_Loss={val_loss:.4f}")


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
#    device = torch.device("cuda:2")
    schedule = NoiseScheduleVP(schedule='cosine')
    writer = SummaryWriter(log_dir="./runs/conditional_diffusion") if local_rank == 0 else None
    print(torch.cuda.get_device_name())

    swin_config = {
      'in_chans': 4,               # 输入通道数：4 个变量
      'embed_dim': 96,             # 初始嵌入维度
      'depths': [2, 2, 6, 2],      # 每层 Swin block 的数量
      'num_heads': [3, 6, 12, 24], # 每层的 attention head 数
      'window_size': (1, 4, 4),  # 空间窗口，时间不切（T=1），H/W 可被432整除
      'patch_size': (1, 4, 4),
        'patch_norm': True,
        'pretrained': None
    }
    encoder = SwinConditionEncoder(swin_config, output_dim=256).to(device)
    unet = CrossAttentionUNet(in_channels=1, cond_dim=256).to(device)

    encoder = DDP(encoder, device_ids=[local_rank])
    unet = DDP(unet, device_ids=[local_rank])
    optimizer = optim.Adam(list(encoder.parameters()) + list(unet.parameters()), lr=1e-4)

    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = 'siconca/abs'

    cmip_names = ['EC-Earth3/r2i1p1f1', 'MRI-ESM2-0/r1i1p1f1']
    cmip_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names)
    cmip_sampler = DistributedSampler(cmip_dataset,shuffle=True, drop_last=True)
    cmip_loader = DataLoader(cmip_dataset, batch_size=4, sampler=cmip_sampler, num_workers=4)
    reanal_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'obs')
    total_len = len(reanal_dataset) 

    from torch.utils.data import Subset
    train_indices = list(range(0, total_len - 156))
    valid_indices = list(range(total_len - 156, total_len - 120))
    test_indices = list(range(total_len - 120, total_len))
    
    # 创建子集
    reanal_train_set = Subset(reanal_dataset, train_indices)
    reanal_valid_set = Subset(reanal_dataset, valid_indices)
    reanal_test_set = Subset(reanal_dataset, test_indices)
    
    # 创建分布式采样器
    reanal_train_sampler = DistributedSampler(reanal_train_set)
    reanal_valid_sampler = DistributedSampler(reanal_valid_set)
    reanal_test_sampler = DistributedSampler(reanal_test_set)
    
    # 创建 DataLoader
    reanal_train_loader = DataLoader(reanal_train_set, batch_size=2, sampler=reanal_train_sampler, num_workers=0)
    reanal_valid_loader = DataLoader(reanal_valid_set, batch_size=2, sampler=reanal_valid_sampler, num_workers=0)
    reanal_test_loader = DataLoader(reanal_test_set, batch_size=2, sampler=reanal_test_sampler, num_workers=0)
    torch.autograd.set_detect_anomaly(True)
    # Pretrain
    train_loop(cmip_loader, reanal_valid_loader,encoder, unet, optimizer, schedule, device, writer, epochs=50, desc="Pretrain", save_path='./model/pretrain.pt')

    if local_rank == 0:
        torch.save({'encoder': encoder.module.state_dict(), 'unet': unet.module.state_dict()}, 'swin_unet_pretrained.pt')

    # Finetune
    train_loop(reanal_train_loader,reanal_valid_loader, encoder, unet, optimizer, schedule, device, writer, epochs=100, desc="Finetune", save_path='./model/finetune_best.pt')
    if local_rank == 0:
        torch.save({'encoder': encoder.module.state_dict(), 'unet': unet.module.state_dict()}, 'swin_unet.pt')

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
