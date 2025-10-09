import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import torch.nn.functional as F
from uvit import UViT
from dataloader import ClimateForecastDataset
from SICForecast import SeaIceForecastPipeline
from SICForecast import NoiseScheduleVP
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def q_sample(x_start, t, noise, schedule):
    alpha = schedule.marginal_mean_coeff(t)
    sigma = schedule.marginal_std(t)
    return (
        alpha[..., None, None, None] * x_start +
        sigma[..., None, None, None] * noise
    )
    
def plot_predictions(y_true, y_pred,x_t, epoch, stage,save_dir="./plotslossmod"):
    os.makedirs(save_dir, exist_ok=True)
    B, T, H, W = y_true.shape

    for i in range(min(3, B)):
        nrows = 3 if x_t is not None else 2
        fig, axes = plt.subplots(nrows, T, figsize=(3 * T, 3 * nrows))

        for t in range(T):
            im_true = axes[0, t].imshow(y_true[i, t].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[0, t].set_title(f"True t={t}")
            axes[0, t].axis('off')
            plt.colorbar(im_true, ax=axes[0, t], fraction=0.046, pad=0.04)

            im_pred = axes[1, t].imshow(y_pred[i, t].detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
            axes[1, t].set_title(f"Pred t={t}")
            axes[1, t].axis('off')
            plt.colorbar(im_pred, ax=axes[1, t], fraction=0.046, pad=0.04)

            if x_t is not None:
                im_xt = axes[2, t].imshow(x_t[i, t].detach().cpu().numpy(), cmap='viridis', vmin=0, vmax=1)
                axes[2, t].set_title(f"Noised t={t}")
                axes[2, t].axis('off')
                plt.colorbar(im_xt, ax=axes[2, t], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stage}_epoch{epoch+1}_sample{i}.png"))
        plt.close()

    
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


def train_loop(dataloader, model, optimizer, schedule, device, epochs, desc):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y in tqdm(dataloader, desc=f"{desc} Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            B, V, T_in, H, W = x.shape
            T_out = y.shape[1]

            # 构造条件输入
            x_cond = x.permute(0, 2, 1, 3, 4).reshape(B, V * T_in, H, W)

            # 构造扩散输入
            noise = torch.randn_like(y)
            # t = torch.rand(B, device=device) * schedule.T
            t_max = schedule.T * min(1.0, (epoch + 1) / epochs)
            t = torch.rand(B, device=device) * t_max
            x_t = q_sample(y, t, noise, schedule)  # 扩散加噪

            x_input = torch.cat([x_cond, x_t], dim=1)
            pred_noise = model(x_input, t)  # 模型预测噪声

            # 损失：噪声预测 + 预测 x0 的分布对齐
            loss_denoise = F.mse_loss(pred_noise, noise)

            # 推回 x0_pred，用于 histogram loss
            alpha = schedule.marginal_mean_coeff(t)
            sigma = schedule.marginal_std(t)
            x0_pred = (x_t - sigma[..., None, None, None] * pred_noise) / torch.exp(alpha[..., None, None, None])

            loss_hist = histogram_kl_loss(x0_pred, y)
            loss = 1 * loss_denoise + 0.1 * loss_hist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #if epoch == epochs - 1 and desc.lower() == "pretrain":
        plot_predictions(y, x0_pred,x_t, epoch, desc)
        print("pred_noise mean:", pred_noise.mean().item(), "std:", pred_noise.std().item())
        print("x0_pred mean:", x0_pred.mean().item(), "std:", x0_pred.std().item())
        print(f"{desc} Epoch {epoch+1}: MSE: {loss_denoise.item():.4f}, Hist: {loss_hist.item():.4f}")
        writer.add_scalar(f"{desc}/MSE", loss_denoise.item(), epoch)
        writer.add_scalar(f"{desc}/KL", loss_hist.item(), epoch)
        writer.add_scalar(f"{desc}/pred_noise_mean", pred_noise.mean().item(), epoch)
        writer.add_scalar(f"{desc}/pred_noise_std", pred_noise.std().item(), epoch)
        writer.add_scalar(f"{desc}/x0_pred_mean", x0_pred.mean().item(), epoch)
        writer.add_scalar(f"{desc}/x0_pred_std", x0_pred.std().item(), epoch)
        
        
if __name__ == '__main__':
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedule = NoiseScheduleVP(schedule='linear')
    writer = SummaryWriter(log_dir="./runs/diffusion_exp")

    EPOCHS_PRETRAIN = 100
    EPOCHS_FINETUNE = 300
    LR = 1e-4
    SAVE_PATH = './icenet_diffusion100er.pt'

    model = UViT(img_size=432, patch_size=16, in_chans=30)
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom' , 'tos/anom']
    target_var = 'siconca/abs'

    cmip_names = [
        'EC-Earth3/r2i1p1f1', 'EC-Earth3/r7i1p1f1', 'EC-Earth3/r10i1p1f1',
        'EC-Earth3/r12i1p1f1', 'EC-Earth3/r14i1p1f1',
            'MRI-ESM2-0/r1i1p1f1',
            'MRI-ESM2-0/r2i1p1f1',
            'MRI-ESM2-0/r3i1p1f1',
            'MRI-ESM2-0/r4i1p1f1',
            'MRI-ESM2-0/r5i1p1f1'    ]

    cmip_dataset = ClimateForecastDataset(
        root_dir=root_dir,
        variables=variables,
        target_var=target_var,
        input_seq_len=6,
        output_seq_len=6,
        mode='transfer',
        model_names=cmip_names
    )
    cmip_dataloader = DataLoader(cmip_dataset, batch_size=32, shuffle=True, num_workers=16)

    reanal_dataset = ClimateForecastDataset(
        root_dir=root_dir,
        variables=variables,
        target_var=target_var,
        input_seq_len=6,
        output_seq_len=6,
        mode='obs'
    )
    keep_indices = list(range(len(reanal_dataset) - 120))
    reanal_dataset_trimmed = Subset(reanal_dataset, keep_indices)
    reanal_dataloader = DataLoader(reanal_dataset_trimmed, batch_size=32, shuffle=True, num_workers=8)

    print("\n[Stage 1] Pretraining on CMIP6 data...")
    train_loop(cmip_dataloader, model, optimizer, schedule, device, EPOCHS_PRETRAIN, desc="Pretrain")
    torch.save(model.state_dict(), SAVE_PATH.replace('.pt', '_100epochpretrained.pt'))

    print("\n[Stage 2] Fine-tuning on observational data...")
    for param in model.parameters():
        param.requires_grad = True
    train_loop(reanal_dataloader, model, optimizer, schedule, device, EPOCHS_FINETUNE, desc="Finetune")
    torch.save(model.state_dict(), SAVE_PATH)
    print("\n✅ Training complete. Model saved to", SAVE_PATH)
    writer.close()
