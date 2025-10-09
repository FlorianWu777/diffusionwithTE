import os, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# ====== 你的 VQ-VAE 定义文件 ======
from trainvqvae import VQVAE3D, ClimateForecastDataset

@torch.no_grad()
def plot_vqvae_recon(
    model_path: str,
    dataloader,
    device: torch.device,
    save_dir: str = "./plots_vqvae",
    var_names=None,
    vmax=None            # 若想 Truth 和 Recon 共用色阶，可手动设
):
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 1. 构建模型并加载权重 ----------
    model = VQVAE3D(in_c=4, lat_c=32, K=256).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ---------- 2. 遍历数据 ----------
    for b_idx, (x, _) in enumerate(tqdm(dataloader, desc="Plot VQ-VAE")):
        x = x.to(device)                                     # [B, C, T, H, W]
        z   = model.encode(x)
        z_q, _ = model.quantise(z)
        recon = model.decode(z_q)
        diff  = (recon - x).abs()

        B, C, T, H, W = x.shape
        for s in range(B):
            for t in range(T):
                fig, axes = plt.subplots(3, C, figsize=(3*C, 9), constrained_layout=True)

                for c in range(C):
                    vmin = -vmax if vmax is not None else None

                    # Truth
                    ax = axes[0, c]
                    im = ax.imshow(x[s, c, t].cpu(), vmin=vmin, vmax=vmax, cmap="viridis")
                    ax.set_title(var_names[c] if var_names else f"Var{c}")
                    ax.set_ylabel("Truth"); ax.axis("off")

                    # Recon
                    ax = axes[1, c]
                    ax.imshow(recon[s, c, t].cpu(), vmin=vmin, vmax=vmax, cmap="viridis")
                    ax.set_ylabel("Recon"); ax.axis("off")

                    # |Diff|
                    ax = axes[2, c]
                    ax.imshow(diff[s, c, t].cpu(), cmap="plasma")
                    ax.set_ylabel("AbsDiff"); ax.axis("off")

                fig.colorbar(im, ax=axes[:, 0], shrink=0.6, location="right")
                fname = f"sample{b_idx}_{s}_t{t}.png"
                plt.savefig(os.path.join(save_dir, fname), dpi=150)
                plt.close()


def main():
    # ───────── DDP 初始化 ─────────
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ───────── 数据集（obs 120 timesteps）─────────
    root = "/data/wuhaotian/diffusionDemo/dataset1"
    vars4 = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    ds  = ClimateForecastDataset(root, vars4, vars4, 12, 12, mode="obs")
    idx = list(range(len(ds)-120, len(ds)))
    subset  = Subset(ds, idx)
    sampler = torch.utils.data.distributed.DistributedSampler(subset, shuffle=False)
    loader  = DataLoader(subset, batch_size=2, sampler=sampler, num_workers=2)

    # ─────────  绘图  ─────────
    plot_vqvae_recon(
        model_path="/data/wuhaotian/diffusionDemo/model/vqvae_best.pt",
        dataloader=loader,
        device=device,
        save_dir="./plots_vqvae",
        var_names=vars4,
        vmax=None           # 或手动给个上限使色阶一致
    )


if __name__ == "__main__":
    main()