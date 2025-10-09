# make_stats.py
"""
计算 dataset1/transfer 下多 CMIP 成员 + 多变量 的整体均值 μ 和标准差 σ
保存为 climate_stats.npz，后续 DataSet 读取即可 (x-μ)/σ 标准化。
"""

from __future__ import annotations
import argparse, numpy as np, os, glob

# ----------------------------------------------------------------------
def welford_update(mean, m2, count, batch):
    # batch: [N, ...] float32
    bcount = batch.size
    if bcount == 0:
        return mean, m2, count
    batch_mean = batch.mean()
    batch_var  = (batch.astype(np.float64) - batch_mean).var()
    delta      = batch_mean - mean
    tot_count  = count + bcount
    mean      += delta * bcount / tot_count
    m2        += batch_var * bcount + delta**2 * count * bcount / tot_count
    return mean, m2, tot_count

# ----------------------------------------------------------------------
def accumulate(root, models, var):
    mean, m2, n = 0.0, 0.0, 0
    for model in models:
        pattern = os.path.join(root, model, var, "*.npy")
        for f in glob.iglob(pattern):
            arr   = np.load(f).astype(np.float32, copy=False)  # [H,W] or [T,H,W]
            mean, m2, n = welford_update(mean, m2, n, arr.ravel())
    std = np.sqrt(m2 / (n - 1))
    return mean, std

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/wuhaotian/diffusionDemo/dataset1/transfer",
                        help="transfer 目录的绝对路径")
    parser.add_argument("--vars", nargs="+", default=["psl/anom", "siconca/abs", "tas/anom", "tos/anom"])
    parser.add_argument("--models", nargs="+", default=[
        "EC-Earth3/r2i1p1f1", "EC-Earth3/r7i1p1f1", "EC-Earth3/r10i1p1f1",
        "EC-Earth3/r12i1p1f1", "EC-Earth3/r14i1p1f1",
        "MRI-ESM2-0/r1i1p1f1", "MRI-ESM2-0/r2i1p1f1", "MRI-ESM2-0/r3i1p1f1",
        "MRI-ESM2-0/r4i1p1f1", "MRI-ESM2-0/r5i1p1f1"
    ])
    parser.add_argument("--out", default="climate_stats.npz")
    args = parser.parse_args()

    means, stds = [], []
    for var in args.vars:
        print(f"?  {var} …")
        mu, sigma = accumulate(args.root, args.models, var)
        means.append(mu); stds.append(sigma)
        print(f"   → μ={mu:.4f}  σ={sigma:.4f}")

    np.savez(args.out, mean=np.array(means, dtype=np.float32),
             std=np.array(stds, dtype=np.float32))
    print(f"? Saved to {args.out}")

if __name__ == "__main__":
    main()
