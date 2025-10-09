# decoder_with_sampler.py
import torch, torch.nn.functional as F
import pytorch_lightning as pl
from .plms import PLMSSampler

class DecoderFineTuneWithSampler(pl.LightningModule):
    def __init__(self, autoenc, unet, context_encoder,
                 lr=1e-3, lpips_w=0.05, num_sample_steps=25):
        super().__init__()
        self.save_hyperparameters(ignore=["autoenc", "unet",
                                          "scheduler", "context_encoder"])
        self.autoenc   = autoenc         # 包含 encoder & decoder
        self.contexter = context_encoder.eval()
        self.num_steps = num_sample_steps
        self.unet = unet  # 注册为模型参数
        self.sampler = PLMSSampler(self.unet, schedule="linear")
        self.sampler.make_schedule(ddim_num_steps=self.num_steps, verbose=False)


        # 只训 decoder
        for n, p in self.autoenc.named_parameters():
            p.requires_grad_(n.startswith("decoder"))

        # 简单 LPIPS：VGG16 relu3_3
        from torchvision.models.feature_extraction import create_feature_extractor
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=True)
        #vgg = torch.hub.load("pytorch/vision", "vgg16", pretrained=True)
        self.perc = create_feature_extractor(vgg.eval(),
                                             return_nodes={'features.16': 'r'})
        for p in self.perc.parameters(): p.requires_grad_(False)

        
        self.lpips_w   = lpips_w
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perc = self.perc.to(device)
    # ---------- step ----------
    def training_step(self, batch, _):
        x_past, x_future = batch  # 形状 (B,C,T,H,W)
        B, V, T, H, W = x_past.shape
        device = x_past.device
        t_relative = torch.linspace(0, 1, T, device=x_past.device).unsqueeze(0).repeat(B, 1)  # shape [B, T]

        x_past = [(x_past, t_relative)]
        with torch.no_grad():
            ctx = self.contexter(x_past)
            finer, _, latents, _ = self.autoenc.encode(x_future)
            init_noise = torch.randn_like(latents)
        
            z_hat, _ = self.sampler.sample(
                S=self.num_steps,
                batch_size=latents.shape[0],
                shape=latents.shape[1:],     # 不包含 batch dim
                
                
                conditioning=ctx,
                x_T=init_noise,
                unconditional_guidance_scale=1.0,
                unconditional_conditioning=None
            )

        x_hat = self.autoenc.decoder(finer, z_hat)          # 只这一块要梯度
        l1  = F.l1_loss(x_hat, x_future)
        mse = F.mse_loss(x_hat, x_future)
       # vgg_hat, vgg_tgt = self.perc(x_hat[..., -1]), self.perc(x_future[..., -1])
       # perc = F.mse_loss(vgg_hat, vgg_tgt)

        loss = l1 + mse #+ self.lpips_w * perc
        self.log_dict({"loss": loss, "l1": l1, "mse": mse}, #, "perc": perc},
                      prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.autoenc.decoder.parameters(),
                               lr=self.hparams.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5_000)
        return [opt], [sch]