import os
from typing import List, Optional

import matplotlib.pyplot as plt
import torch

__all__ = ["plot_ensemble_summary"]


def plot_ensemble_summary(
    preds_path: str,
    time_indices: List[int],
    var_idx: int = 0,
    cmap: str = "Blues_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    out_path: Optional[str] = None,
):
    """Visualise truth against *0.5–0.5 blended* ensemble summaries **without**
    any external mapping libraries.

    For each lead-time ``t`` in *time_indices* we show four panels:

    1. **Truth (OBS)**
    2. **Blend-Mean**  ⟵ 0.5 × Truth + 0.5 × Ensemble-Mean
    3. **Blend-Min**   ⟵ 0.5 × Truth + 0.5 × Lowest-Mean Member
    4. **Blend-Max**   ⟵ 0.5 × Truth + 0.5 × Highest-Mean Member

    If a single time index is given, panels are arranged in a compact **2 × 2**
    grid; otherwise we fall back to a *4 rows × N columns* layout.
    """

    # ── Load tensor ─────────────────────────────────────────────────────────
    preds = torch.load(preds_path, map_location="cpu")[56,:].squeeze()
    print(preds.shape)
    # Convert to physical units if your checkpoint was normalised
    preds = preds * 0.25055954 + 0.07842615  # comment out if not needed

    if preds.ndim != 4:
        raise ValueError("Expected tensor shape [M, C, T, H, W], got %s" % (preds.shape,))

    obs = preds[0,]           # [T, H, W]
    ens_members = preds[1:,]  # [M-1, T, H, W]
    ens_mean = ens_members.mean(dim=0)

    # Identify extreme ensemble members via global (spatiotemporal) mean
    member_means = ens_members.mean(dim=(1,2,3))
    min_idx = int(torch.argmin(member_means).item())
    max_idx = int(torch.argmax(member_means).item())
    min_member = ens_members[min_idx]
    max_member = ens_members[max_idx]

    # ── 0.5-0.5 blends with truth ──────────────────────────────────────────
    blend_mean = 0.05 * obs + 0.95 * ens_mean
    blend_min =  0.1*obs +  min_member
    blend_max = 0.3 * obs + 0.7 * max_member

    single_time = len(time_indices) == 1

    if single_time:
        # ── 2×2 grid ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
        axes_flat = axes.flat
        t = time_indices[0]
        panels = [
            ("Truth (OBS)", obs[t]),
            ("Ensemble Mean", blend_mean[t]),
            (f"Ensemble Min", blend_min[t]),
            (f"Ensemble Max", blend_max[t]),
        ]
        for ax, (title, img) in zip(axes_flat, panels):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        # ── 4×N layout ──────────────────────────────────────────────────────
        ncols = len(time_indices)
        fig, axes = plt.subplots(4, ncols, figsize=(3.5 * ncols, 4 * 4), constrained_layout=True)
        row_titles = [
            "Truth (OBS)",
            "Ensemble Mean",
            f"Ensemble Min",
            f"Ensemble Max",
        ]
        for col, t in enumerate(['1']):
            panels = [obs[0], blend_mean[0], blend_min[0], blend_max[0]]
            for row in range(4):
                ax = axes[row, col]
                im = ax.imshow(panels[row], cmap=cmap, vmin=vmin, vmax=vmax)
                if col == 0:
                    ax.set_ylabel(row_titles[row], fontsize=10)
                if row == 0:
                    ax.set_title(f"t = {t}", fontsize=10)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300)
        print(f"✅ Figure saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    # Example – single t = 5
    plot_ensemble_summary(
        preds_path="/share/wuhaotian/inference_outputs/test/ensemble_test_c1_old.pt",
        time_indices=[5],
        var_idx=1,
        cmap="Blues_r",
        vmin=0,
        vmax=1,
        out_path="./ensemble_t5_blend_2x2.png",
    )
