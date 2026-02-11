import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module, ABC):
    def __init__(self, betas, device):
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
    
    
    @abstractmethod
    def p_sample(self, model, x, t):
        raise NotImplementedError()
    
    @abstractmethod
    def p_sample_loop(self, model, shape):
        raise NotImplementedError()
    

class DDPMScheduler(GaussianDiffusion):
    def __init__(self, betas, device, variance_type='fixed_small'):
        super().__init__(betas, device)

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
        pred_noise = model(x, t)   
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
    
    
    def p_sample(self, model, x, t, **kwargs):
        """t 시점에서의 샘플링 (p(x_{t-1}|x_t))

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


class DDIMScheduler(GaussianDiffusion):
    def __init__(self, betas, device, eta=0.0):
        super().__init__(betas, device)
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
    
    
    def p_sample(self, model, x, t, t_idx, clip_denoised=True):
        pred_noise = model(x, t)
        
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
    

    def p_sample_loop(self, model, shape, clip_denoised=True, sampling_steps=50):
        self._set_timesteps(sampling_steps)
        
        device = self.betas.device
        batch_size = shape[0]
        
        img = torch.randn(shape, device=device)
        
        for i in range(sampling_steps):
            t = torch.full((batch_size,), self.timesteps[i], device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i, clip_denoised)
        return img        




if __name__ == "__main__":
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_model = lambda x, t: torch.randn_like(x)
    
    ddpm_scheduler = DDPMScheduler(betas, device)
    ddpm_scheduler.to(device)   

    ddpm_scheduler.p_sample_loop(model=dummy_model, shape=(16, 3, 32, 32)) 
    print("DDPM schedulers test passed.")
    
    ddim_scheduler = DDIMScheduler(betas, device, eta=0.0)
    
    ddim_scheduler.to(device)
    ddim_scheduler.p_sample_loop(model=None, shape=(16, 3, 32, 32), sampling_steps=50)
    print("Diffusion schedulers test passed.")