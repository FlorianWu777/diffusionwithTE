import torch
from torch import nn
import pytorch_lightning as pl

from ..distributions import kl_from_standard_normal, ensemble_nll_normal
from ..distributions import sample_from_standard_normal


class ResBlock3D(nn.Module):
    def __init__(
        self, in_channels, out_channels, resample=None,
        resample_factor=(1,1,1), kernel_size=(3,3,3), 
        act='swish', norm='group', norm_kwargs=None, 
        spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        
        padding = tuple(k//2 for k in kernel_size)
        if resample == "down":
            self.resample = nn.AvgPool3d(resample_factor, ceil_mode=True)       
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                kernel_size=kernel_size, stride=resample_factor, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
        elif resample == "up":
            self.resample = nn.Upsample(
                scale_factor=resample_factor, mode='trilinear')            
            self.conv1 = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
            output_padding = tuple(
                2*p+s-k for (p,s,k) in zip(padding,resample_factor,kernel_size)
            )
            self.conv2 = nn.ConvTranspose3d(out_channels, out_channels,
                kernel_size=kernel_size, stride=resample_factor,
                padding=padding, output_padding=output_padding)
        else:
            self.resample = nn.Identity()
            self.conv1 = nn.Conv3d(in_channels, out_channels,
                kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size=kernel_size, padding=padding)

        if isinstance(act, str):
            act = (act, act)
        self.act1 = activation(act_type=act[0])
        self.act2 = activation(act_type=act[1])

        if norm_kwargs is None:
            norm_kwargs = {}
        self.norm1 = normalization(in_channels, norm_type=norm, **norm_kwargs)
        self.norm2 = normalization(out_channels, norm_type=norm, **norm_kwargs)
        if spectral_norm:
            self.conv1 = sn(self.conv1)
            self.conv2 = sn(self.conv2)
            if not isinstance(self.proj, nn.Identity):
                self.proj = sn(self.proj)


    def forward(self, x):
        x_in = self.resample(self.proj(x))
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x + x_in


class SimpleConvEncoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64):
        sequence = []
        channels = np.hstack([
            in_dim, 
            (8**np.arange(1,levels+1)).clip(min=min_ch)
        ])
        
        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i+1])
            res_kernel_size = (3,3,3) if i == 0 else (1,3,3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            downsample = nn.Conv3d(out_channels, out_channels,
                kernel_size=(2,2,2), stride=(2,2,2))
            sequence.append(downsample)
            in_channels = out_channels

        super().__init__(*sequence)


class SimpleConvDecoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=2, min_ch=64):
        sequence = []
        channels = np.hstack([
            in_dim, 
            (8**np.arange(1,levels+1)).clip(min=min_ch)
        ])

        for i in reversed(list(range(levels))):
            in_channels = int(channels[i+1])
            out_channels = int(channels[i])
            upsample = nn.ConvTranspose3d(in_channels, in_channels, 
                    kernel_size=(2,2,2), stride=(2,2,2))
            sequence.append(upsample)
            res_kernel_size = (3,3,3) if (i == 0) else (1,3,3)
            res_block = ResBlock3D(
                in_channels, out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1}
            )
            sequence.append(res_block)
            in_channels = out_channels

        super().__init__(*sequence)



class AutoencoderKL(pl.LightningModule):
    def __init__(
        self, 
        encoder, decoder, 
        kl_weight=0.01,     
        encoded_channels=64,
        hidden_width=32,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width
        self.to_moments = nn.Conv3d(encoded_channels, 2*hidden_width,
            kernel_size=1)
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels,
            kernel_size=1)
        self.log_var = nn.Parameter(torch.zeros(size=()))
        self.kl_weight = kl_weight

    def encode(self, x):
        h = self.encoder(x)
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        return (mean, log_var)

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        (mean, log_var) = self.encode(input)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        dec = self.decode(z)
        return (dec, mean, log_var)

    def _loss(self, batch):
        (x,y) = batch
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0][0]
        (y_pred, mean, log_var) = self.forward(x)

        rec_loss = (y-y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)

        total_loss = rec_loss + self.kl_weight * kl_loss

        return (total_loss, rec_loss, kl_loss)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)[0]
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        (total_loss, rec_loss, kl_loss) = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}_loss", total_loss, **log_params)
        self.log(f"{split}_rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}_kl_loss", kl_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val_rec_loss",
                "frequency": 1,
            },
        }
