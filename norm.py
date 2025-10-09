#!/usr/bin/env python3
"""
Normalize EC-Earth3 siconca .npy grids:
– 对文件取值范围先做检查，若 max > 1 则整体 ÷100 并覆盖保存；
– 若 max ≤ 1 则认为已归一化，跳过。
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm   # pip install tqdm

BASE_DIR = Path("/share/wuhaotian/dataset1/transfer/MRI-ESM2-0")
R_NUMS   = [1, 2, 3, 4, 5]
YEAR_MIN, YEAR_MAX = 1850, 2100

def normalize_file(fp: Path) -> None:
    """Load → check max → maybe divide by 100 → save回原文件."""
    try:
        arr = np.load(fp, allow_pickle=False)
    except Exception as e:
        print(f"[ERROR] 读取失败 {fp}: {e}")
        return

    vmax = float(arr.max())
    if vmax > 1.1:                       # 只有没归一化的才处理
        arr = (arr / 100.0).astype(np.float32)
        np.save(fp, arr)
        print(f"[OK]  归一化完成: {fp.name:>12s} (max {vmax:.1f} → {arr.max():.2f})")
    else:
        print(f"[SKIP] 已归一化:  {fp.name:>12s} (max {vmax:.2f})")

def main():
    for r in R_NUMS:
        run_dir = BASE_DIR / f"r{r}i1p1f1" / "siconca" / "abs"
        if not run_dir.exists():
            print(f"[WARN] 路径不存在: {run_dir}")
            continue

        # 遍历年份与月份；仅处理实际存在的文件
        for yyyy in range(YEAR_MIN, YEAR_MAX + 1):
            for mm in range(1, 13):
                fp = run_dir / f"{yyyy}_{mm:02d}.npy"
                if fp.is_file():
                    normalize_file(fp)

if __name__ == "__main__":
    # tqdm 进度条包裹整个循环，方便查看整体进度
    for r in R_NUMS:
        run_dir = BASE_DIR / f"r{r}i1p1f1" / "siconca" / "abs"
        if not run_dir.exists():
            print(f"[WARN] 路径不存在: {run_dir}")
            continue

        year_month_paths = [
            run_dir / f"{yyyy}_{mm:02d}.npy"
            for yyyy in range(YEAR_MIN, YEAR_MAX + 1)
            for mm in range(1, 13)
            if (run_dir / f"{yyyy}_{mm:02d}.npy").is_file()
        ]

        print(f"\n=== r{r}i1p1f1: 发现 {len(year_month_paths)} 个文件 ===")
        for fp in tqdm(year_month_paths, desc=f"r{r}i1p1f1 归一化"):
            normalize_file(fp)
