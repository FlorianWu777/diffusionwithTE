from __future__ import annotations
from typing import Optional

import einops
import torch
from torch import nn

__all__ = ["KernelCRPS", "AlmostFairKernelCRPS"]

# --------------------------------------------------
# 1. 极简 BaseLoss：只提供 ignore_nans + reduce
# --------------------------------------------------
class _BaseLoss(nn.Module):
    def __init__(self, ignore_nans: bool = False) -> None:
        super().__init__()
        self._mean = torch.nanmean if ignore_nans else torch.mean
        self._sum  = torch.nansum  if ignore_nans else torch.sum

    def _reduce(
        self,
        out: torch.Tensor,
        squash: bool = True,
        squash_mode: str = "avg",
        *,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        """(bs, ens, grid, var) → 标量"""
        if squash:
            if squash_mode == "avg":
                out = self._mean(out, dim=-1)          # var 维
            elif squash_mode == "sum":
                out = self._sum(out,  dim=-1)
            else:
                raise ValueError("squash_mode must be 'avg' or 'sum'")

        # grid 维求和，batch & ensemble 求平均
      #  out = self._mean(self._sum(out, dim=2), dim=(0,1))
        out = self._mean(out, dim=(0, 1, 2)) 
        if group is not None:
            torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=group)
            out /= torch.distributed.get_world_size(group)
        return out


# --------------------------------------------------
# 2. Kernel-/Almost-Fair CRPS
#    期望输入：
#      y_pred   : (bs, ens, grid, var)
#      y_target : (bs,       grid, var)
# --------------------------------------------------
class KernelCRPS(_BaseLoss):
    def __init__(self, fair: bool = True, ignore_nans: bool = False) -> None:
        super().__init__(ignore_nans)
        self.fair = fair

    # ---------- 核心公式 ----------
    @staticmethod
    def _kernel(preds: torch.Tensor, target: torch.Tensor, fair: bool) -> torch.Tensor:
        # preds  : [bs, e, g, v]   target: [bs, g, v]
        bs, e, g, v = preds.shape
        mae = torch.mean(torch.abs(preds - target.unsqueeze(1)), dim=1)     # [bs,g,v]

        coef = -1.0 / (e*(e-1)) if fair else -1.0 / (e*e)
        # 上三角 pairwise |xi-xj|
        ens_var = 0.
        for i in range(e - 1):
            ens_var += torch.sum(torch.abs(preds[:, i].unsqueeze(1) - preds[:, i+1:]), dim=1)
        ens_var = coef * ens_var                                            # [bs,g,v]

        return mae + ens_var                                                # [bs,g,v]

    # ---------- 前向 ----------
    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        squash: bool = True,
        *,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> torch.Tensor:
        if y_pred.ndim != 4 or y_target.ndim != 3:
            raise ValueError("y_pred shape (bs,ens,grid,var), y_target (bs,grid,var)")

        diff = self._kernel(y_pred, y_target, self.fair).unsqueeze(1)  # → (bs,1,grid,var)
        return self._reduce(diff, squash=squash, group=group)

    @property
    def name(self) -> str:
        return ("f" if self.fair else "") + "kcrps"
