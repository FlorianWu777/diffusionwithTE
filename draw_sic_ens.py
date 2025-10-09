import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_members_sorted_by_sie(
    preds_path,
    time_indices,
    sample_idx=0,
    var_idx=0,
    out_path=None,
    cmap="Blues_r",
    vmin=None,
    vmax=None,
    sie_threshold=0.15,
):
    """
    4x5 子图：
    - 第 1 张为 OBS（标注自身 SIE）
    - 其余 19 张为成员，按平均 SIE 排序，标题标注成员 SIE 与 IoU。
    """
    # ==== 加载并反标准化 ====
    preds = torch.load(preds_path, map_location="cpu")  # [B, C, T, H, W]
    assert preds.ndim == 5, f"Expected [B,C,T,H,W], got {preds.shape}"
    preds = preds * 0.25055954 + 0.07842615

    B, C, T, H, W = preds.shape
    assert 0 <= var_idx < C, "var_idx 越界"
    assert len(time_indices) > 0, "需要至少一个 time index"
    time_indices = [t for t in time_indices if 0 <= t < T]
    assert len(time_indices) > 0, "time_indices 全部越界"

    # 展示的时刻
    t_show = time_indices[0]

    # ==== 取 OBS 与成员 ====
    obs_field = preds[0, var_idx, t_show]            # [H, W]
    member_fields_show = preds[1:, var_idx, t_show]  # [M, H, W], M=B-1

    # ==== 计算排序所用的平均 SIE（在给定 time_indices 上平均）====
    with torch.no_grad():
        members_over_times = preds[1:, var_idx, time_indices]   # [M, K, H, W]
        mask = (members_over_times > sie_threshold).to(torch.int32)
        sie_counts = mask.sum(dim=(2, 3)).float()               # [M, K]
        sie_avg = sie_counts.mean(dim=1)                        # [M]

    # 成员索引：按 SIE 从小到大（最多 19 个）
    sort_idx = torch.argsort(sie_avg).tolist()[:19]

    # ==== 准备绘图（4x5）====
    nrows, ncols = 4, 5
    total_panels = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows))
    axes = axes.flatten()

    # 统一色标范围
    if vmin is None or vmax is None:
        members_to_plot = member_fields_show[sort_idx] if len(sort_idx) > 0 else member_fields_show
        stack = torch.stack([obs_field] + [m for m in members_to_plot], dim=0)
        if vmin is None: vmin = float(stack.min().item())
        if vmax is None: vmax = float(stack.max().item())

    # ==== OBS 面板 ====
    obs_mask = (obs_field > sie_threshold)
    obs_sie_cells = int(obs_mask.sum().item())

    im0 = axes[0].imshow(obs_field, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"OBS | SIE={obs_sie_cells} cells", fontsize=11)
    axes[0].axis("off")

    # ==== 成员面板 ====
    for k in range(1, total_panels):
        ax = axes[k]
        if k-1 < len(sort_idx):
            m_idx = sort_idx[k-1]
            field = member_fields_show[m_idx]
            m_mask = (field > sie_threshold)

            # 成员 SIE
            m_sie = int(m_mask.sum().item())

            # IoU
            inter = torch.logical_and(m_mask, obs_mask).sum().item()
            union = torch.logical_or(m_mask, obs_mask).sum().item()
            iou = 1.0 if union == 0 else inter / union

            ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Member {m_idx+1} | SIE={m_sie} | IoU={iou:.3f}", fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")


    plt.suptitle(
        f"SIC | Members sorted by SIE over Leadtime={time_indices} months",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"✅ 图像保存到：{out_path}")

    plt.show()


# 用法示例
plot_members_sorted_by_sie(
    preds_path="/share/wuhaotian/inference_outputs/test/test_idx00000.pt",
    time_indices=[5],
    sample_idx=5,
    var_idx=1,
    cmap="Blues_r",
    vmin=0,
    vmax=1,
    out_path="./seaice_preds_sorted_4x5_sie_iou.png",
    sie_threshold=0.15,
)
