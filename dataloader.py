import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Sequence, Union, Optional
from pathlib import Path



class QuarterResolution:
    """将气候场的空间分辨率缩减到原来的 1/4（长宽各减半）。"""

    def __init__(self, scale: float = 0.5, mode: str = "bilinear"):
        if not (0 < scale <= 1):
            raise ValueError("scale must be in (0, 1]")
        self.scale = scale
        self.mode = mode

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """支持形状为 [V, T, H, W] 或 [C, T, H, W] 的张量。"""
        if tensor.dim() != 4:
            raise ValueError(f"Expected tensor with 4 dims, got {tensor.shape}")

        v, t, h, w = tensor.shape
        flat = tensor.reshape(v * t, 1, h, w)

        align = None
        if self.mode in {"bilinear", "bicubic"}:
            align = False

        down = F.interpolate(flat, scale_factor=self.scale, mode=self.mode, align_corners=align)
        new_h, new_w = down.shape[-2:]
        return down.view(v, t, new_h, new_w)



class ClimateForecastDataset1(Dataset):
    def __init__(
        self,
        root_dir: str,
        variables: Sequence[str],
        target_var: Sequence[str],
        input_seq_len: int = 12,
        output_seq_len: int = 6,
        mode: str = "transfer",
        model_names: Optional[Union[str, Sequence[str]]] = None,
        transform=None,
    ):
        assert mode in {"transfer", "obs"}
        # 1) 处理 model_names → List[str]
        if model_names is None:
            model_names = []
        if isinstance(model_names, str):
            model_names = [model_names]
        self.model_names: List[str] = list(model_names)

        if mode == "transfer" and not self.model_names:
            # 未指定就默认扫全部子目录
            self.model_names = sorted(
                d for d in os.listdir(os.path.join(root_dir, mode))
                if os.path.isdir(os.path.join(root_dir, mode, d))
            )

        # 2) 每个 model 一个根目录
        self.root_dirs = [
            os.path.join(root_dir, mode, m) for m in self.model_names
        ] if mode == "transfer" else [os.path.join(root_dir, mode)]

        self.variables    = list(variables)
        self.target_var   = list(target_var)
        self.in_len       = input_seq_len
        self.out_len      = output_seq_len
        self.transform    = transform

        # 3) 先算“单模型内部交集”，再算“跨模型交集”
        self.common_files = self._get_common_file_list()

        # 4) 预先记录“每个模型可用的时间步数”
        self.model_sample_len = (
            len(self.common_files) - self.in_len - self.out_len + 1
        )
        assert self.model_sample_len > 0, "时间序列太短"

    # -------------------------------------------------------------- #
    def _files_for(self, root, var):
        p = os.path.join(root, var)
        return sorted(f for f in os.listdir(p) if f.endswith(".npy"))

    def _get_common_file_list(self):
        per_model_sets = []
        for r in self.root_dirs:
            var_sets = []
            for v in set(self.variables):
                var_sets.append(set(self._files_for(r, v)))
            per_model_sets.append(set.intersection(*var_sets))
        return sorted(list(set.intersection(*per_model_sets)))
    # -------------------------------------------------------------- #

    # 5) “所有模型 × 时间窗” 展平成一条长序列
    def __len__(self):
        return self.model_sample_len * len(self.root_dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        # -------- 映射到具体 (model_idx, time_idx) --------
        model_idx  = idx // self.model_sample_len
        time_idx   = idx %  self.model_sample_len
        mroot      = self.root_dirs[model_idx]

        in_files   = self.common_files[time_idx :
                                       time_idx + self.in_len]
        out_files  = self.common_files[time_idx + self.in_len :
                                       time_idx + self.in_len + self.out_len]

        # -------- 构造输入 [V, T_in, H, W] --------
        inputs = []
        for var in self.variables:
            seq = [np.load(os.path.join(mroot, var, f)) for f in in_files]
            inputs.append(seq)
        inputs = torch.from_numpy(np.array(inputs)).float()

        # -------- 构造输出 [T_out, H, W] --------
        targets = []
        for var in self.target_var:
            seq = [np.load(os.path.join(mroot, var, f)) for f in out_files]
            targets.append(seq)
        targets = torch.from_numpy(np.array(targets)).float()
        return inputs, targets


class ClimateForecastDataset(Dataset):
    """
    支持多 CMIP 模型 (transfer) / 单 obs 目录 (obs)。
    返回:
        x: [V, T_in, H, W]  float32
        y: [T_out, H, W]    float32   (若 target_var>1 可改成 [V_tgt, T_out, H, W])
    """
    def __init__(
        self,
        root_dir: str,
        variables: Sequence[str],
        target_var: Sequence[str],
        input_seq_len: int = 12,
        output_seq_len: int = 6,
        mode: str = "transfer",              # 'transfer' 或 'obs'
        model_names: Optional[Union[str, Sequence[str]]] = None,
        dtype=np.float32,                    # 一次性控制精度
        transform=None,
    ):
        assert mode in {"transfer", "obs"}
        self.dtype = dtype
        # ---------- 1. 解析 model_names ----------
        if model_names is None:
            model_names = []
        if isinstance(model_names, str):
            model_names = [model_names]

        if mode == "transfer" and not model_names:
            model_names = sorted(
                d for d in os.listdir(Path(root_dir, mode))
                if Path(root_dir, mode, d).is_dir()
            )
        self.root_dirs = [Path(root_dir, mode, *m.split('/')) for m in model_names]
        
        if mode == "obs":
            model_names = Path(root_dir, 'obs')
            self.root_dirs = [model_names] 
        
            
        self.variables   = list(variables)
        self.target_var  = list(target_var)
        self.target_var = self.variables
        self.in_len      = input_seq_len
        self.out_len     = output_seq_len
        self.transform   = transform

        # ---------- 2. 求交集文件名 ----------
        self.common_files = self._calc_common_files()
        self.samples_per_model = len(self.common_files) - self.in_len - self.out_len + 1
        assert self.samples_per_model > 0, "时间序列不足以滑窗"

    # ---------------- 私有工具 -----------------
    def _files_in(self, root: Path, var: str):
        path = root.joinpath(*var.split('/'))
        print(f"Checking path: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Variable path does not exist: {path}")
        return sorted(f.name for f in path.iterdir() if f.suffix == ".npy")


    def _calc_common_files(self) -> List[str]:
        per_model_sets = []
        for r in self.root_dirs:
            print(f"\n Model dir: {r}")
            var_sets = []
            for v in set(self.variables + self.target_var):
                files = set(self._files_in(r, v))

                var_sets.append(files)
            common = set.intersection(*var_sets)

            per_model_sets.append(common)
        
        if not per_model_sets:
            raise RuntimeError("No valid variable files found across models.")
        
        total_common = set.intersection(*per_model_sets)

        return sorted(total_common)
    # ------------------------------------------

    def __len__(self):
        return self.samples_per_model * len(self.root_dirs)

    def _load_seq(self, root: Path, var: str, files: Sequence[str]):
        arr = [np.load(root / var / f).astype(self.dtype, copy=False) for f in files]
        return np.stack(arr, axis=0)             # [T, H, W]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        model_idx = idx // self.samples_per_model
        t0        = idx  % self.samples_per_model
        root      = self.root_dirs[model_idx]

        in_files  = self.common_files[t0 : t0 + self.in_len]
        out_files = self.common_files[t0 + self.in_len : t0 + self.in_len + self.out_len]

        # -------- 输入 --------
        x = [self._load_seq(root, v, in_files) for v in self.variables]
        x = torch.from_numpy(np.stack(x, axis=0))        # [V, T_in, H, W]

        # -------- 输出 --------
        y = [self._load_seq(root, v, out_files) for v in self.target_var]
        y = torch.from_numpy(np.stack(y, axis=0))        # [V_tgt, T_out, H, W]

        # 若只想返回 target 单变量 → y = y[0]

        if self.transform:
            x, y = self.transform(x), self.transform(y)
        return x.float(), y.float()
class ClimateForecastDataset_wotransfer(Dataset):
    def __init__(self, root_dir, variables, target_var, input_seq_len=12, output_seq_len=6, mode='transfer',  model_name=None, transform=None):
        """
        root_dir: 数据根目录 (/data/diffusionDemo/dataset1)
        variables: 输入变量列表，如 ['psl/anom', 'siconca/abs', ...]
        target_var: 预测目标变量名，如 'siconca/abs'
        input_seq_len: 输入序列长度（过去月份数）
        output_seq_len: 输出序列长度（未来月份数）
        mode: 'transfer' for pretrain, 'obs' for finetuning
        transform: 可选的数据变换操作
        """
        self.mode = mode
        self.model_name = model_name if mode == 'transfer' else 'obs'
        self.root_dir = os.path.join(root_dir, mode)
        if self.model_name=='transfer':
            self.root_dir = os.path.join(self.root_dir, self.model_name)

        self.variables = variables
        self.target_var = target_var
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.transform = transform

        self.file_list = self._get_sorted_file_list()

    def _get_sorted_file_list(self):
        """获取所有变量文件的交集，并排序以确保时间顺序"""
        file_sets = []
        for var in set(self.variables + [self.target_var]):
            var_dir = os.path.join(self.root_dir, var)
            files = sorted([f for f in os.listdir(var_dir) if f.endswith('.npy')])
            file_sets.append(set(files))

        common_files = sorted(list(set.intersection(*file_sets)))
        return common_files

    def __len__(self):
        # 减去输入和输出序列长度，防止索引溢出
        return len(self.file_list) - self.input_seq_len - self.output_seq_len + 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_files = self.file_list[idx:idx + self.input_seq_len]
        output_files = self.file_list[idx + self.input_seq_len:idx + self.input_seq_len + self.output_seq_len]

        input_data = []
        for var in self.variables:
            var_seq = []
            for file in input_files:
                file_path = os.path.join(self.root_dir, var, file)
                var_seq.append(np.load(file_path))
            input_data.append(var_seq)

        # 加载输出目标变量
        output_data = []
        for var in self.variables:
            var_seq = []
            for file in output_files:
                file_path = os.path.join(self.root_dir, self.target_var, file)
                var_seq.append(np.load(file_path))
            output_data.append(var_seq)

        input_data = np.array(input_data)  # shape: [num_variables, input_seq_len, 432, 432]
        output_data = np.array(output_data)  # shape: [output_seq_len, 432, 432]
        
        input_data = torch.from_numpy(input_data).float()
        output_data = torch.from_numpy(output_data).float()

        if self.transform:
            input_data = self.transform(input_data)
            output_data = self.transform(output_data)

        return input_data, output_data


# 使用示例
if __name__ == "__main__":
    root_dir = '/data/wuhaotian/diffusionDemo/dataset1'
    variables = ['psl/anom', 'siconca/abs', 'tas/anom' , 'tos/anom']  # 替换为所有所需变量名，共12个
    target_var = ['siconca/abs']
    cmip_names = ['EC-Earth3/r2i1p1f1',
                    'EC-Earth3/r7i1p1f1',
                    'EC-Earth3/r10i1p1f1',
                    'EC-Earth3/r12i1p1f1',
                    'EC-Earth3/r14i1p1f1',
                    'MRI-ESM2-0/r1i1p1f1',
                    'MRI-ESM2-0/r2i1p1f1',
                    'MRI-ESM2-0/r3i1p1f1',
                    'MRI-ESM2-0/r4i1p1f1',
                    'MRI-ESM2-0/r5i1p1f1']
                    
    dataset = ClimateForecastDataset(
        root_dir=root_dir,
        variables=variables,
        target_var=target_var,
        model_names = cmip_names,
        input_seq_len=12,
        output_seq_len=12,
        mode='transfer'
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    # 测试 DataLoader
    for inputs, targets in dataloader:
        print('Input shape:', inputs.shape)   # [batch_size, num_variables, 12, 432, 432]
        print('Target shape:', targets.shape) # [batch_size, 6, 432, 432]
        break
