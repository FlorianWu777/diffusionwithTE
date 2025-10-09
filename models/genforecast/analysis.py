import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from ..nowcast.nowcast import AFNONowcastNetBase
from ..blocks.resnet import ResBlock3D


class AFNONowcastNetCascade_raw(AFNONowcastNetBase):
    def __init__(self, *args, cascade_depth=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.cascade_depth = cascade_depth
        self.resnet = nn.ModuleList()        
        ch = self.embed_dim_out
        self.cascade_dims = [ch]
        for i in range(cascade_depth-1):
            ch_out = 2*ch
            self.cascade_dims.append(ch_out)
            self.resnet.append(
                ResBlock3D(ch, ch_out, kernel_size=(1,3,3), norm=None)
            )
            ch = ch_out

    def forward(self, x):
        x = super().forward(x)
        img_shape = tuple(x.shape[-2:])
        cascade = {img_shape: x}
        for i in range(self.cascade_depth-1):
            x = F.avg_pool3d(x, (2,3,3))
            x = self.resnet[i](x)
            img_shape = tuple(x.shape[-2:])
            cascade[img_shape] = x
        return cascade

class AFNONowcastNetCascade(nn.Module):
    """
    一个包装器，将 AFNO 模型与年循环时间嵌入（TE）结合起来。
    """
    def __init__(self, afno_model: nn.Module, te_model: nn.Module):
        super().__init__()
        self.afno = afno_model
        self.te = te_model

    def forward(self, x: torch.Tensor, month_idx: Optional[torch.Tensor] = None):
        """
        前向传播流程：
        1. 首先对输入 x 应用时间嵌入。
        2. 然后将带有时间信息的张量送入 AFNO 模型。
        """
    def forward(self, context: list, month_idx: Optional[torch.Tensor] = None):
        # 1. 从 context 列表中解包
        x_tensor, t_rel = context[0]
        
        # 2. 对张量应用时间嵌入
        x_with_te = self.te(x_tensor, month_idx=month_idx)
        
        # 3. 关键修复：将处理后的张量与 t_rel 重新打包成基类期望的格式
        repackaged_context = [(x_with_te, t_rel)]
        
        # 4. 将【重新打包好】的 context 传递给核心 AFNO 模型
        output = self.afno(repackaged_context)
        
        return output
    def __getattr__(self, name: str):
        """
        属性委托：如果在此包装器上找不到属性，
        则自动从内部的 self.afno 模型中查找。
        这使得访问 afno 内部的属性（如 embed_dim_out）变得无缝。
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.afno, name):
                return getattr(self.afno, name)
            raise