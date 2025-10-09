import torch
import numpy as np
from tqdm import tqdm

from .utils import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

class DDIMSampler:
    def __init__(self, model, schedule="linear",eta=1):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.eta = eta
    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.2, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose
        )

        alphas_cumprod = self.model.alphas_cumprod
        device = next(self.model.parameters()).device

        to_torch = lambda x: x.clone().detach().to(torch.float32).to(device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=self.eta,
            verbose=verbose
        )

        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, eta=0.0, verbose=False, progbar=False, **kwargs): # from eta=0.2 to eta=0.0
        #print('eta',eta)
        self.make_schedule(ddim_num_steps=S, ddim_eta=self.eta, verbose=verbose)
        size = (batch_size,) + shape
        #print(f"[DDIM] Using seed: {torch.initial_seed()}")  # 打印当前global seed
        img = torch.randn(size, device=self.model.betas.device)
        #print(f"[DDIM] Initial img std: {img.std().item():.5f}, mean: {img.mean().item():.5f}")
        intermediates = {'x_inter': [img]}

        timesteps = np.flip(self.ddim_timesteps)

        iterator = tqdm(timesteps, desc='DDIM Sampler', disable=not progbar)

        for i, step in enumerate(iterator):
            ts = torch.full((batch_size,), step, device=img.device, dtype=torch.long)
            index = len(timesteps) - i - 1

            img, pred_x0 = self.p_sample_ddim(img, conditioning, ts, index)

            intermediates['x_inter'].append(img)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index):
        device = x.device
    
        e_t = self.model.apply_model(x, t, c)
      
        alpha       = torch.tensor(self.ddim_alphas[index], device=device, dtype=torch.float32)
        alpha_prev  = torch.tensor(self.ddim_alphas_prev[index], device=device, dtype=torch.float32)
        sigma       = torch.tensor(self.ddim_sigmas[index], device=device, dtype=torch.float32)
        sqrt_one_minus_alpha = torch.tensor(self.ddim_sqrt_one_minus_alphas[index], device=device, dtype=torch.float32)
    
          # 确保所有变量都为 torch.float32，并在 GPU 上
        alpha       = alpha.to(device).float()
        alpha_prev  = alpha_prev.to(device).float()
        sigma       = sigma.to(device).float()
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.to(device).float()
    
        pred_x0 = (x - sqrt_one_minus_alpha * e_t) / torch.sqrt(alpha)
    
        safe_term = torch.clamp(1. - alpha_prev - sigma ** 2, min=0.0)
        dir_xt = torch.sqrt(safe_term) * e_t
    
        noise = sigma * torch.randn_like(x)
        x_prev = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise
    
        return x_prev, pred_x0
