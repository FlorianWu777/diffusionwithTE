# vqvae_latent_diffusion.py  (patched)

from __future__ import annotations
import os, numpy as np, torch
from pathlib import Path
from typing import Sequence, List, Union, Optional, Tuple

import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pytorch_lightning as pl
import re
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset with built-in (x-μ)/σ standardisation
# ─────────────────────────────────────────────────────────────────────────────
_STATS_CACHE: dict[str, dict[str, np.ndarray]] = {}

def load_stats(path: str):
    if path not in _STATS_CACHE:
        d = np.load(path)
        _STATS_CACHE[path] = {"mean": d["mean"].astype("float32"),
                              "std":  d["std" ].astype("float32")}
    return _STATS_CACHE[path]

DEFAULT_VARS  = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
DEFAULT_CMIP = [
    'EC-Earth3/r2i1p1f1','EC-Earth3/r7i1p1f1','EC-Earth3/r10i1p1f1',
    'EC-Earth3/r12i1p1f1','EC-Earth3/r14i1p1f1',
    'MRI-ESM2-0/r1i1p1f1','MRI-ESM2-0/r2i1p1f1','MRI-ESM2-0/r3i1p1f1',
    'MRI-ESM2-0/r4i1p1f1','MRI-ESM2-0/r5i1p1f1'
]

class ClimateForecastDataset(Dataset):
    """CMIP/obs .npy dataset with μ/σ scaling, with month metadata support."""
    def __init__(
        self,
        root_dir: str,
        variables: Optional[Sequence[str]] = None,
        target_var: Optional[Sequence[str]] = None,
        input_seq_len: int = 12,
        output_seq_len: int = 6,
        mode: str = "transfer",
        model_names: Optional[Union[str, Sequence[str]]] = None,
        stats_path: Optional[str] = None,
        dtype=np.float32,
        return_meta: bool = False,        # ★ 新增：是否返回月份等元信息
        spatial_downsample: float = 1.0,
        downsample_mode: str = "bilinear",
    ):
        assert mode in {"transfer", "obs"}
        self.dtype = dtype
        self.return_meta = return_meta
        self.spatial_downsample = float(spatial_downsample)
        self.downsample_mode = downsample_mode
        if self.spatial_downsample <= 0:
            raise ValueError("spatial_downsample must be positive")

        if stats_path is None:
            stats_path = os.path.join(root_dir, "climate_stats.npz")
        self.stats = load_stats(stats_path)

        # ----- model list --------------------------------------------------
        if model_names is None:
            model_names = []
        if isinstance(model_names, str):
            model_names = [model_names]
        if mode == "transfer" and not model_names:
            model_names = DEFAULT_CMIP

        self.root_dirs = ([Path(root_dir, mode, m) for m in model_names]
                          if mode == "transfer" else [Path(root_dir, mode)])

        self.variables  = list(variables or DEFAULT_VARS)
        self.target_var = list(target_var or self.variables)
        self.in_len, self.out_len = input_seq_len, output_seq_len

        # ----- intersect timestamps ---------------------------------------
        self.common_files = self._intersect_files()
        self.samples_per_model = len(self.common_files) - self.in_len - self.out_len + 1
        assert self.samples_per_model > 0, "time series too short"

        mean = torch.tensor(self.stats["mean"])[:, None, None, None]
        std  = torch.tensor(self.stats["std"])[:,  None, None, None]
        self.mean = mean          # torch.tensor([V,1,1,1])
        self.std  = std

    # ---------------------------------------------------------------------
    def _files_in(self, root: Path, var: str) -> List[str]:
        return sorted(f.name for f in (root / var).iterdir() if f.suffix == ".npy")

    def _intersect_files(self) -> List[str]:
        per_model = []
        for r in self.root_dirs:
            var_sets = [set(self._files_in(r, v)) for v in set(self.variables + self.target_var)]
            per_model.append(set.intersection(*var_sets))
        return sorted(set.intersection(*per_model))

    def __len__(self) -> int:
        return self.samples_per_model * len(self.root_dirs)

    def _load_seq(self, root: Path, var: str, fns: List[str]) -> np.ndarray:
        return np.stack([np.load(root / var / f).astype(self.dtype, copy=False) for f in fns])

    # ====== 月份解析相关 ===================================================
    _PATTERNS = [
        # 优先匹配 YYYY[-_]?MM[-_]?DD
        re.compile(r"(?P<y>\d{4})[-_]?((?P<m>0[1-9]|1[0-2]))[-_]?(\d{2})"),
        # 其次 YYYY[-_]?MM
        re.compile(r"(?P<y>\d{4})[-_]?((?P<m>0[1-9]|1[0-2]))"),
    ]
    _MM_FALLBACK = re.compile(r"(?:^|[^0-9])(?P<m>0[1-9]|1[0-2])(?:[^0-9]|$)")

    @staticmethod
    def _stem(fn: str) -> str:
        return os.path.splitext(fn)[0]

    def _month_from_filename(self, fn: str) -> Optional[int]:
        """尽力从文件名里解析月份（1..12）。失败则返回 None。"""
        stem = self._stem(fn)
        for pat in self._PATTERNS:
            m = pat.search(stem)
            if m:
                mm = int(m.group("m"))
                if 1 <= mm <= 12:
                    return mm
        # 兜底：找任何独立出现的 01..12
        m2 = self._MM_FALLBACK.search(stem)
        if m2:
            mm = int(m2.group("m"))
            if 1 <= mm <= 12:
                return mm
        return None

    def _months_from_files(self, fns: List[str]) -> Optional[torch.LongTensor]:
        """逐帧解析月份；若有任一失败，返回 None。"""
        months: List[int] = []
        for f in fns:
            mm = self._month_from_filename(f)
            if mm is None:
                return None
            months.append(mm)
        return torch.tensor(months, dtype=torch.long)

    @staticmethod
    def _roll_months_from_start(m0: int, T: int) -> torch.LongTensor:
        """从起始月份 m0（1..12）生成长度 T 的月份序列，跨年取模。"""
        arr = [((m0 - 1 + k) % 12) + 1 for k in range(T)]
        return torch.tensor(arr, dtype=torch.long)
    
    def debug_filenames(self, idx: int):
        """返回给定样本的输入/输出文件名列表与根路径（不读数据）"""
        if torch.is_tensor(idx):
            idx = idx.item()
        samples_per_model = self.samples_per_model
        model_idx = idx // samples_per_model
        t0        = idx  % samples_per_model
        root      = self.root_dirs[model_idx]
    
        in_f  = self.common_files[t0 : t0 + self.in_len]
        out_f = self.common_files[t0 + self.in_len : t0 + self.in_len + self.out_len]
        return in_f, out_f, root
    

    # ======================================================================
    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()
        model_idx = idx // self.samples_per_model
        t0        = idx  % self.samples_per_model
        root      = self.root_dirs[model_idx]

        in_f  = self.common_files[t0 : t0 + self.in_len]
        out_f = self.common_files[t0 + self.in_len : t0 + self.in_len + self.out_len]

        x = torch.from_numpy(np.stack([self._load_seq(root, v, in_f)  for v in self.variables]))
        y = torch.from_numpy(np.stack([self._load_seq(root, v, out_f) for v in self.target_var]))

        x = (x - self.mean) / (self.std + 1e-6)
        y = (y - self.mean[: y.size(0)]) / (self.std[: y.size(0)] + 1e-6)

        x = self._downsample(x)
        y = self._downsample(y)

        if not self.return_meta:
            return x.float(), y.float()

        # ---------- 构造 meta：优先逐帧解析；否则“从首帧推全段” ----------
        months_in  = self._months_from_files(in_f)
        months_out = self._months_from_files(out_f)

        if months_in is None:
            m0 = self._month_from_filename(in_f[0])
            months_in = self._roll_months_from_start(m0 if m0 is not None else 1, self.in_len)
        if months_out is None:
            m0o = self._month_from_filename(out_f[0])
            months_out = self._roll_months_from_start(m0o if m0o is not None else 1, self.out_len)

        meta: Dict[str, object] = {
            "month_in":  months_in,    # torch.LongTensor[T_in], 1..12
            "month_out": months_out,   # torch.LongTensor[T_out], 1..12
            "start_file_in": in_f[0],  # 便于排查
        }
        return x.float(), y.float(), meta

    def _downsample(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.spatial_downsample == 1.0:
            return tensor.float()

        if tensor.ndim != 4:
            raise ValueError("Expected tensor with shape [variables, timesteps, H, W]")

        v, t, h, w = tensor.shape
        reshaped = tensor.reshape(v * t, 1, h, w)
        kwargs = {}
        if self.downsample_mode in {"linear", "bilinear", "bicubic", "trilinear"}:
            kwargs["align_corners"] = False
        resized = F.interpolate(
            reshaped,
            scale_factor=self.spatial_downsample,
            mode=self.downsample_mode,
            **kwargs,
        )
        new_h, new_w = resized.shape[-2:]
        return resized.reshape(v, t, new_h, new_w)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    target_var = ['psl/anom', 'siconca/abs', 'tas/anom', 'tos/anom']
    
    cmip_names = ['EC-Earth3/r2i1p1f1', 'MRI-ESM2-0/r1i1p1f1']
    cmip_dataset = ClimateForecastDataset(root_dir, variables, target_var, 12, 1, 'transfer', cmip_names)
    #cmip_sampler = DistributedSampler(cmip_dataset, shuffle=True, drop_last=True)
    #cmip_loader = DataLoader(cmip_dataset, batch_size=4, sampler=cmip_sampler, num_workers=4)
    for i in range(13):
        in_f, out_f, root = cmip_dataset.debug_filenames(i)
        print(root)
        print(in_f)
        print(out_f)


if __name__ == "__main__":
    main()
