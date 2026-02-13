import torch
from models.Diffusion.schedulers.gaussian_diffusion import GaussianDiffusion


class DDIMScheduler(GaussianDiffusion):
    def __init__(self, betas, device, eta=0.0, null_token_idx=None):
        super().__init__(betas, device, null_token_idx)
        self.eta = eta
    
    
    def _set_timesteps(self, sampling_steps):
        self.sampling_timesteps = sampling_steps
        self.timesteps = torch.linspace(0, len(self.betas) - 1, sampling_steps).long().flip(dims=[0])
    
        self.alpha_cumprod_at_t = self.alphas_cumprod[self.timesteps]
        self.alpha_cumprod_at_prev = torch.cat(
            [self.alpha_cumprod_at_t[1:], torch.tensor([1.0], device=self.betas.device)], dim=0
        )
        
        at_t = self.alpha_cumprod_at_t
        at_prev = self.alpha_cumprod_at_prev
        
        if self.eta == 0.0:
            self.sigma_at_t = torch.zeros_like(at_t)
        else:
            self.sigma_at_t = self.eta * torch.sqrt(
                (1 - at_prev) / (1 - at_t) * (1 - at_t / at_prev)
            )
    
    def p_sample(self, model, x, t, t_idx, model_kwargs=None, guidance_scale=1.0, clip_denoised=True):
        if guidance_scale > 1.0 and model_kwargs is not None:
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            
            uncond_kwargs = {
                k: torch.full_like(v, self.null_token_idx) 
                for k, v in model_kwargs.items()
            }
            
            combined_kwargs = {
                k: torch.cat([v, uncond_kwargs[k]], dim=0)
                for k, v in model_kwargs.items()
            }
            model_out = model(x_in, t_in, **combined_kwargs)
            out_cond, out_uncond = model_out.chunk(2, dim=0)
            pred_noise = out_uncond + guidance_scale * (out_cond - out_uncond)
        else:
            pred_noise = model(x, t, **(model_kwargs or {}))
        
        at_t = self.alpha_cumprod_at_t[t_idx]
        at_prev = self.alpha_cumprod_at_prev[t_idx]
        sigma_t = self.sigma_at_t[t_idx]

        sqrt_recip_at_t = torch.sqrt(1.0 / at_t)
        sqrt_one_minus_at_t = torch.sqrt(1.0 - at_t)
        
        pred_x0 = sqrt_recip_at_t * x - (sqrt_recip_at_t * sqrt_one_minus_at_t) * pred_noise   
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
        dir_xt = torch.sqrt(1.0 - at_prev - sigma_t**2) * pred_noise
        
        if sigma_t > 0:
            random_noise = torch.randn_like(x)
        else:
            random_noise = 0.0
        
        return torch.sqrt(at_prev) * pred_x0 + dir_xt + sigma_t * random_noise   
    

    def p_sample_loop(self, model, shape, model_kwargs=None, guidance_scale=1.0, sampling_steps=50):
        self._set_timesteps(sampling_steps)
        
        device = self.betas.device
        batch_size = shape[0]
        
        img = torch.randn(shape, device=device)
        
        for i in range(sampling_steps):
            t = torch.full((batch_size,), self.timesteps[i], device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i, model_kwargs=model_kwargs, guidance_scale=guidance_scale)
        return img        