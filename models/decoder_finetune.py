# decoder_with_sampler.py
import torch, torch.nn.functional as F
import pytorch_lightning as pl
from .plms import PLMSSampler 

class DecoderFineTuneWithSampler(pl.LightningModule):
    def __init__(self, autoenc, unet, scheduler, context_encoder,
                 lr=1e-4, lpips_w=0.05, num_sample_steps=25):
        super().__init__()
        self.save_hyperparameters(ignore=["autoenc", "unet",
                                          "scheduler", "context_encoder"])
        self.autoenc   = autoenc          # 包含 encoder & decoder
        self.contexter = context_encoder.eval()
        self.sampler = PLMSSampler(unet, schedule="linear")
        self.sampler.make_schedule(ddim_num_steps=self.num_steps)

        # 只训 decoder
        for n, p in self.autoenc.named_parameters():
            p.requires_grad_(n.startswith("decoder"))

        # 简单 LPIPS：VGG16 relu3_3
        from torchvision.models.feature_extraction import create_feature_extractor
        vgg = torch.hub.load("pytorch/vision", "vgg16", pretrained=True)
        self.perc = create_feature_extractor(vgg.eval(),
                                             return_nodes={'features.16': 'r'})
        for p in self.perc.parameters(): p.requires_grad_(False)

        self.num_steps = num_sample_steps
        self.lpips_w   = lpips_w

    # ---------- step ----------
    def training_step(self, batch, _):
        x_past, x_future = batch["past"].float(), batch["future"].float()  # 形状 (B,C,T,H,W)
        with torch.no_grad():
            ctx = self.contexter(x_past)
            latents = self.autoenc.encode(x_future).latent_dist.mode()
            init_noise = torch.randn_like(latents)
        
            z_hat, _ = self.sampler.sample(
                S=self.num_steps,
                batch_size=latents.shape[0],
                shape=latents.shape[1:],        # 不包括 batch
                conditioning=ctx,
                x_T=init_noise,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None
            )
        x_hat = self.autoenc.decoder(z_hat)          # 只这一块要梯度
        l1  = F.l1_loss(x_hat, x_future)
        mse = F.mse_loss(x_hat, x_future)
        vgg_hat, vgg_tgt = self.perc(x_hat[..., -1]), self.perc(x_future[..., -1])
        perc = F.mse_loss(vgg_hat, vgg_tgt)

        loss = l1 + mse + self.lpips_w * perc
        self.log_dict({"loss": loss, "l1": l1, "mse": mse, "perc": perc},
                      prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.autoenc.decoder.parameters(),
                               lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5_000)
        return [opt], [sch]