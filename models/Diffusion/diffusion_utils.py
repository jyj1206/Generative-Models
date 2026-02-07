import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    def __init__(self, betas, device, variance_type='fixed_small'):
        super().__init__()
        
        betas = betas.to(device)
        self.register_buffer('betas', betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)

        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)

        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
            
        if variance_type == 'fixed_small':
            posterior_variance_t = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        else:
            posterior_variance_t = betas[1:]
            
        posterior_variance_t = torch.cat(
            [torch.tensor([0.0], device=self.betas.device), posterior_variance_t], dim=0
        )
        
        self.register_buffer('posterior_variance', posterior_variance_t)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance_t, min=1e-20)))
        

    def _extract(self, a, t, x_shape): 
        """t 시점의 값 가져오기

        Args:
            a : 1D Tensor (timesteps,)
            t : timesteps (Batch,)
            x_shape (tuple): Shape of the target tensor

        Returns:
            out : Extracted Tensor (Batch, 1, 1, 1)
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    
    def q_sample(self, x_start, t, noise=None):
        """t 시점의 noisy sample 생성
        
        Args:
            x_start : Original Image (Batch, C, H, W)
            t : timesteps (Batch,)
            noise : Optional Noise Tensor (Batch, C, H, W)

        Returns:
            out : Noisy Image at timestep t (Batch, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    
    def p_losses(self, model, x_start, t, noise=None):
        """time step t에서의 손실 계산
        
        Args:
            model : The noise prediction model
            x_start : Original Image (Batch, C, H, W)
            t : timesteps (Batch,)
            noise : Optional Noise Tensor (Batch, C, H, W)

        Returns:
            loss : MSE loss between true noise and predicted noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = model(x_noisy, t)
        return F.mse_loss(noise, pred_noise)
    
    
    def p_mean_variance(self, model, x, t, clip_denoised=True):
        """t 시점의 posterior의 평균 분산 계산

        Args:
            model : The noise prediction model
            x : Noisy Image at timestep
            t : timesteps (Batch,)
            clip_denoised: clip the denoised image into [-1, 1]

        Returns:
            model_mean : Mean of the posterior
            model_log_variance : Log Variance of the posterior
        """
        pred_noise = model(x, t)   
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        pred_x0 = sqrt_recip_alphas_t * x - (sqrt_recip_alphas_t * sqrt_one_minus_alphas_cumprod) * pred_noise
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x.shape) *pred_x0 + 
            self._extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        
        model_log_variance = self._extract(self.posterior_log_variance_clipped, t, x.shape)

        return model_mean, model_log_variance
    
    
    def p_sample(self, model, x, t):
        """t 시점에서의 샘플링

        Args:
            model: The noise prediction model
            x: Noisy Image at time
            t: timesteps (Batch,)

        Returns:
            out: Sampled Image at timestep t-1
        """
        model_mean, model_log_variance = self.p_mean_variance(model, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = (t > 0).float().view(-1, *((1,) * (len(x.shape) - 1)))  
        return model_mean + torch.exp(0.5 * model_log_variance) * noise * nonzero_mask
    
    
    def p_sample_loop(self, model, shape):
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
            img = self.p_sample(model, img, t)
        
        return img


if __name__ == "__main__":
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    model = GaussianDiffusion(betas=betas, device='cpu')
    print(model)
    
    sample_shape = (4, 3, 32, 32)
    samples = model.p_sample_loop(model=lambda x, t: x, shape=sample_shape)
    
    print("Generated samples shape:", samples.shape)
    assert samples.shape == sample_shape
    
    print("Diffusion model test passed.")
    