import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from trainvqvae import ClimateForecastDataset
from simpleautoencoder import ConvEncoder3D, ConvDecoder3D, AutoencoderKL  # 修改为你模型定义文件的路径

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.no_grad()
def plot_autoencoder_recon(
    model_or_state,
    dataloader,
    device: torch.device,
    save_dir: str = "./plots",
    var_names=None,        # 长度=4 的变量名列表
    vmax=None              # 手动控制色阶上限 (对称区间)
):
    """
    为 dataloader 中的每个 sample 在 12 个时间步上分别绘图。
    输出一张 3×4 子图：Truth / Recon / AbsDiff × 4 个变量(channel)。

    Args
    ----
    model_path : str         已保存的 .pt 权重文件
    dataloader : DataLoader  生成形状 [B, C(=4), T(=12), H, W] 的批
    device      : torch.device
    save_dir    : str        输出目录
    var_names   : list[str]  ['psl/anom', 'siconca/abs', ...]
    vmax        : float|None 若给定则 Truth 与 Recon 使用 [-vmax, vmax]
    """
    if isinstance(model_or_state, torch.nn.Module):
        model = model_or_state
    elif isinstance(model_or_state, dict):
        # 说明是 state_dict，需要先构建模型
        encoder = ConvEncoder3D(in_channels=4, latent_channels=32).to(device)
        decoder = ConvDecoder3D(latent_channels=32, out_channels=4).to(device)
        model = AutoencoderKL(
            encoder=encoder,
            decoder=decoder,
            kl_weight=0.01,
            encoded_channels=32,
            hidden_width=32,
        ).to(device)
        state = remove_module_prefix(model_or_state)
        model.load_state_dict(state)
    else:
        raise TypeError("Unsupported input type for model/state")

    model.eval()

    # ---------- 2. 遍历数据 ----------
    for b_idx, (x, _) in enumerate(tqdm(dataloader, desc="Plotting")):
        x = x.to(device)                         # [B, C, T, H, W]
        recon = model(x, sample_posterior=False)[0]

        diff = (recon - x).abs()                 # [B, C, T, H, W]

        # ---------- 3. 逐 sample, 逐 T ----------
        B, C, T, H, W = x.shape
        for s in range(B):
            for t in range(T):
                fig, axes = plt.subplots(
                    nrows=3, ncols=C, figsize=(3 * C, 9),
                    constrained_layout=True
                )
                vmin = 0
                vmax = 1

                # 上下 3 行：真值 / 重建 / 误差
                for c in range(C):
                    vmin = -vmax if vmax is not None else None
                    # Truth
                    ax = axes[0, c]
                    im = ax.imshow(
                        x[s, c, t].cpu(), vmin=vmin, vmax=vmax, cmap="viridis"
                    )
                    ax.set_title(var_names[c] if var_names else f"Var{c}")
                    ax.set_ylabel("Truth")
                    ax.axis("off")

                    # Recon
                    ax = axes[1, c]
                    ax.imshow(
                        recon[s, c, t].cpu(), vmin=vmin, vmax=vmax, cmap="viridis"
                    )
                    ax.set_ylabel("Recon")
                    ax.axis("off")

                    # |Diff|
                    ax = axes[2, c]
                    ax.imshow(
                        diff[s, c, t].cpu(), cmap="plasma"
                    )
                    ax.set_ylabel("AbsDiff")
                    ax.axis("off")

                # 可选：给第一列加 colorbar
                fig.colorbar(im, ax=axes[:, 0], shrink=0.6, location="right")
                fname = f"sample{b_idx}_{s}_t{t}.png"
                plt.savefig(os.path.join(save_dir, fname), dpi=150)
                plt.close()



def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print('eval')
    # === 模型构建 ===
    encoder = ConvEncoder3D(in_channels=4, latent_channels=32).to(device)
    decoder = ConvDecoder3D(latent_channels=32, out_channels=4).to(device)
    model = AutoencoderKL(
        encoder=encoder,
        decoder=decoder,
        kl_weight=0.003,
        encoded_channels=32,
        hidden_width=32
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # === 加载权重 ===
    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    state_dict = torch.load("/data/wuhaotian/diffusionDemo/model/finetuned_autoencoder.pt", map_location=map_location)
    model.module.load_state_dict(state_dict)
    model.eval()

    # === 构建测试数据集 ===
    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 12, 'obs')

    total_len = len(dataset)
    test_indices = list(range(total_len - 120, total_len))
    test_subset = Subset(dataset, test_indices)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_subset, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=2, sampler=test_sampler, num_workers=2)

    # === 获取一个批次进行可视化 ===
    plot_autoencoder_recon(
        model_or_state=model,
        dataloader=test_loader,
        device=device,
        save_dir="./plots",
        var_names=variables,
        vmax=None         # 如果希望 Truth/Recon 共享同一色阶，可手动设置
    )


if __name__ == "__main__":
    main()
