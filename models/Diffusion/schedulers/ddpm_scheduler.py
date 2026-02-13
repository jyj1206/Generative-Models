import torch
import torch.nn.functional as F
from models.Diffusion.schedulers.gaussian_diffusion import GaussianDiffusion


class DDPMScheduler(GaussianDiffusion):
    def __init__(self, betas, device, variance_type='fixed_small', null_token_idx=None):
        super().__init__(betas, device, null_token_idx)

        betas = self.betas
            
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.betas.device), self.alphas_cumprod[:-1]], dim=0)

        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(self.alphas_cumprod_prev) * betas / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(self.alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
            
        if variance_type == 'fixed_small':
            posterior_variance_t = betas[1:] * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        else:
            posterior_variance_t = betas[1:]
            
        posterior_variance_t = torch.cat(
            [torch.tensor([0.0], device=self.betas.device), posterior_variance_t], dim=0
        )
        
        self.register_buffer('posterior_variance', posterior_variance_t)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance_t, min=1e-20)))
    
    
    def p_mean_variance(self, x, t, pred_noise, clip_denoised=True):
        """t 시점의 q(x_t|x_t, x_0)의 posterior의 평균, 분산 계산하여, 
        학습된 모델의 reverse process 분포 p(x_{t-1}|x_t) 추정 

        Args:
            model : The noise prediction model
            x : Noisy Image at timestep
            t : timesteps (Batch,)
            clip_denoised: clip the denoised image into [-1, 1]

        Returns:
            model_mean : Mean of the posterior
            model_log_variance : Log Variance of the posterior
        """
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        pred_x0 = sqrt_recip_alphas_t * x - (sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod) * pred_noise
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) * pred_x0 + 
            self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        
        model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)

        return model_mean, model_log_variance
    
    
    def p_sample(self, model, x, t, model_kwargs=None, guidance_scale=1.0):
        """t 시점에서의 샘플링 (p(x_{t-1}|x_t))

        Args:
            model: The noise prediction model
            x: Noisy Image at time
            t: timesteps (Batch,)

        Returns:
            out: Sampled Image at timestep t-1
        """
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
        
        model_mean, model_log_variance = self.p_mean_variance(x, t, pred_noise)
        noise = torch.randn_like(x)
        nonzero_mask = (t > 0).float().view(-1, *((1,) * (len(x.shape) - 1)))  
        return model_mean + torch.exp(0.5 * model_log_variance) * noise * nonzero_mask
    
    
    def p_sample_loop(self, model, shape, model_kwargs=None, guidance_scale=1.0):
        """전체 샘플링 루프

        Args:
            model : The noise prediction model
            shape : Shape of the generated samples (Batch, C, H, W)

        Returns:
            img : Generated Image (Batch, C, H, W)
        """
        device = self.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(len(self.betas))):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, model_kwargs=model_kwargs, guidance_scale=guidance_scale)
        
        return img