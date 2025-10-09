from vaeautoencoder import AutoencoderKL
from vaeblock import SimpleConvEncoder, SimpleConvDecoder
import os
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataloader import ClimateForecastDataset

def kl_from_standard_normal(mean, log_var):
    kl = 0.5 * (log_var.exp() + mean.square() - 1.0 - log_var)
    return kl.mean()


def train_autoencoder_loop(dataloader, valid_loader, autoencoder_model, optimizer, device, writer, epochs, desc, save_path):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        dataloader.sampler.set_epoch(epoch)
        autoencoder_model.train()

        for batch_idx, (x, _) in enumerate(tqdm(dataloader, desc=f"[{desc}] Epoch {epoch+1}", disable=dist.get_rank() != 0)):
            x = x.to(device)  # [B, V, T_in, H, W]

            # Forward pass
            (x_recon, mean, log_var) = autoencoder_model(x)

            # Compute losses
            rec_loss = (x - x_recon).abs().mean()
            kl_loss = kl_from_standard_normal(mean, log_var)
            loss = rec_loss + autoencoder_model.module.kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if dist.get_rank() == 0:
            writer.add_scalar(f"{desc}/Recon_Loss", rec_loss.item(), epoch)
            writer.add_scalar(f"{desc}/KL_Loss", kl_loss.item(), epoch)
            print(f"[{desc}] Epoch {epoch+1}: Recon={rec_loss.item():.4f}, KL={kl_loss.item():.4f}")

            # 验证阶段
            val_loss = evaluate_autoencoder(valid_loader, autoencoder_model, device)
            writer.add_scalar(f"{desc}/Val_Loss", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(autoencoder_model.module.state_dict(), save_path)
                print(f"✅ Best Autoencoder saved at epoch {epoch+1} with Val_Loss={val_loss:.4f}")

def evaluate_autoencoder(valid_loader, autoencoder_model, device):
    autoencoder_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, _ in valid_loader:
            x = x.to(device)
            (x_recon, _, _) = autoencoder_model(x)
            rec_loss = (x - x_recon).abs().mean()
            total_loss += rec_loss.item()

    return total_loss / len(valid_loader)
    
    

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{local_rank}")

writer = SummaryWriter(log_dir="./runs/autoencoder") if local_rank == 0 else None

encoder = SimpleConvEncoder(in_dim=4, levels=2, min_ch=64).to(device)
decoder = SimpleConvDecoder(in_dim=4, levels=2, min_ch=64).to(device)
autoencoder_model = AutoencoderKL(
    encoder=encoder,
    decoder=decoder,
    kl_weight=0.01,
    encoded_channels=64,
    hidden_width=32,
).to(device)

autoencoder_model = DDP(autoencoder_model, device_ids=[local_rank],find_unused_parameters=True)

optimizer = optim.AdamW(autoencoder_model.parameters(), lr=1e-4, weight_decay=1e-3)

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

# Train the Autoencoder
train_autoencoder_loop(
    cmip_loader,
    reanal_valid_loader,
    autoencoder_model,
    optimizer,
    device,
    writer,
    epochs=50,
    desc="Autoencoder Training",
    save_path='./model/autoencoder_best.pt'
)

finetune_opt = optim.AdamW(autoencoder_model.parameters(), lr=1e-4, weight_decay=1e-3)
train_autoencoder_loop(
    reanal_train_loader,
    reanal_valid_loader,
    autoencoder_model,
    optimizer,
    device,
    writer,
    epochs=50,
    desc="Autoencoder Training",
    save_path='./model/autoencoder_best.pt'
)
if local_rank == 0:
    torch.save(autoencoder_model.module.state_dict(), 'autoencoder_final.pt')

if writer:
    writer.close()
dist.destroy_process_group()
    


