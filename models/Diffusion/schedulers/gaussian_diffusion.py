import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module, ABC):
    def __init__(self, betas, device, null_token_idx=None):
        super().__init__()
        
        betas = betas.to(device) 
        self.register_buffer('betas', betas)
        
        # for classifier-free guidance
        self.null_token_idx = null_token_idx

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
        """t 시점의 noisy sample 생성 (q(x_t|x_0))
        
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
    

    def p_losses(self, model, x_start, t, noise=None, model_kwargs=None):
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
        
        if model_kwargs is not None:
            pred_noise = model(x_noisy, t, **model_kwargs)
        else:
            pred_noise = model(x_noisy, t)
            
        return F.mse_loss(noise, pred_noise)
    
    
    @abstractmethod
    def p_sample(self, model, x, t, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def p_sample_loop(self, model, shape):
        raise NotImplementedError()