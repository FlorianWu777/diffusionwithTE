"""Dataset utilities for low-resolution diffusion training."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataloader import ClimateForecastDataset1


def _downsample_spatial(
    tensor: torch.Tensor, scale_factor: float, mode: str = "bilinear"
) -> torch.Tensor:
    """Downsample the spatial dimensions of a tensor while preserving leading dims."""
    if scale_factor == 1.0:
        return tensor

    if tensor.ndim < 3:
        raise ValueError("Tensor must have at least 3 dimensions for spatial downsampling")

    height, width = tensor.shape[-2:]
    reshaped = tensor.reshape(-1, 1, height, width)
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        resized = F.interpolate(reshaped, scale_factor=scale_factor, mode=mode, align_corners=False)
    else:
        resized = F.interpolate(reshaped, scale_factor=scale_factor, mode=mode)
    new_h, new_w = resized.shape[-2:]
    return resized.reshape(*tensor.shape[:-2], new_h, new_w)


class DownsampledClimateForecastDataset(Dataset):
    """Wraps :class:`ClimateForecastDataset1` to reduce spatial resolution before returning tensors."""

    def __init__(
        self,
        root_dir: str,
        variables: Sequence[str],
        target_var: Sequence[str],
        input_seq_len: int = 12,
        output_seq_len: int = 12,
        mode: str = "obs",
        model_names: Optional[Union[str, Sequence[str]]] = None,
        scale_factor: float = 0.25,
        downsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.downsample_mode = downsample_mode

        self.base_dataset = ClimateForecastDataset1(
            root_dir=root_dir,
            variables=variables,
            target_var=target_var,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            mode=mode,
            model_names=model_names,
        )

        sample_x, sample_y = self.base_dataset[0]

        sample_x = _downsample_spatial(sample_x, self.scale_factor, self.downsample_mode)
        sample_y = _downsample_spatial(sample_y, self.scale_factor, self.downsample_mode)

        self.low_res_shape: Tuple[int, int] = sample_y.shape[-2:]
        self.input_variable_count = sample_x.shape[0]
        self.input_seq_len = sample_x.shape[1]
        if sample_y.ndim == 4:
            self.output_variable_count = sample_y.shape[0]
            self.output_seq_len = sample_y.shape[1]
        else:
            self.output_variable_count = 1
            self.output_seq_len = sample_y.shape[0]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        x = _downsample_spatial(x, self.scale_factor, self.downsample_mode)
        y = _downsample_spatial(y, self.scale_factor, self.downsample_mode)
        return x.float(), y.float()


__all__ = ["DownsampledClimateForecastDataset"]
