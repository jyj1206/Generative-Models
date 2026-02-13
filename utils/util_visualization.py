import os
import torch
from matplotlib import pyplot as plt
from utils.util_makegif import SampleRecorder
from utils.util_paths import get_output_dir
from utils.utils_images import save_single_image, save_grid_image


def save_loss_curve(configs, loss_history1, loss_history2=None,
                         name1="Train Loss", name2="Test Loss"):
    
    has_second_loss = loss_history2 is not None and len(loss_history2) > 0
    
    num_plots = 2 if has_second_loss else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5)) 

    if num_plots == 1:
        axes = [axes]

    axes[0].plot(loss_history1, label=name1, color='blue')
    axes[0].set_title(name1)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    if has_second_loss:
        axes[1].plot(loss_history2, label=name2, color='orange')
        axes[1].set_title(name2)
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True)

    plt.tight_layout()

    output_dir = get_output_dir(configs)
    save_dir = os.path.join(output_dir, "visualization")
    os.makedirs(save_dir, exist_ok=True) 
    
    save_path = os.path.join(save_dir, "loss_curve.png")

    plt.savefig(save_path)
    print("save loss curve:", save_path)
    plt.close() 


def generate_and_save_samples(model, configs, device, diffusion=None, num_samples=16, epoch=None, scale=4, use_cfg=False, class_id=None, guidance_scale=None, sampling_steps=None, save_dir=None, filename=None, save_to_file=True):
    task = configs.get("task")
    
    with torch.no_grad():
        if task == "vae":
            z = torch.randn(num_samples, model.latent_dim).to(device)
            samples = model(z)
            if configs["model"]['activation'] == 'tanh':
                samples = (samples + 1) / 2
            samples = samples.clamp(0, 1).cpu()
            default_filename = f"generated_samples_epoch_{epoch}.png" if epoch else "generated_samples_final.png"
            normalize = False
            
        elif task == "gan":
            z = torch.randn(num_samples, model.latent_dim).to(device)
            samples = model(z)
            if configs["model"]['activation'] == 'tanh':
                samples = (samples + 1) / 2
            samples = samples.clamp(0, 1).cpu()
            default_filename = f"generated_samples_epoch_{epoch}.png" if epoch else "generated_samples_final.png"
            normalize = False
            
        elif task == "diffusion":
            if diffusion is None:
                raise ValueError("Diffusion scheduler is required for diffusion task.")
            
            model_kwargs = None
            if guidance_scale is None:
                guidance_scale = float(configs.get("diffusion", {}).get("guidance_scale", 1.0))
                
            if use_cfg:
                if class_id is None:
                    num_classes = int(configs.get("dataset", {}).get("num_classes", 1000))
                    class_id = int(torch.randint(0, num_classes, (1,), device=device).item())
                model_kwargs = {"y": torch.full((num_samples,), class_id, device=device, dtype=torch.long)}
            
            sample_shape = (num_samples, configs["model"]['in_channels'], configs["model"]['img_size'], configs["model"]['img_size'])
            if hasattr(diffusion, "_set_timesteps"):
                if sampling_steps is None:
                    sampling_steps = int(configs.get("diffusion", {}).get("sampling_steps", 50))
                samples = diffusion.p_sample_loop(
                    model,
                    shape=sample_shape,
                    sampling_steps=sampling_steps,
                    model_kwargs=model_kwargs,
                    guidance_scale=guidance_scale
                )
            else:
                samples = diffusion.p_sample_loop(
                    model,
                    shape=sample_shape,
                    model_kwargs=model_kwargs,
                    guidance_scale=guidance_scale
                )
            samples = samples.cpu()
            default_filename = f"diffusion_generated_samples_epoch_{epoch}.png" if epoch else "diffusion_generated_samples_final.png"
            normalize = True
            
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    # Save to file if requested
    if save_to_file:
        # Determine save path
        if save_dir is None:
            output_dir = get_output_dir(configs)
            if epoch is None:
                save_dir = os.path.join(output_dir, "visualization")
            else:
                save_dir = os.path.join(output_dir, "visualization", "train")
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename if filename else default_filename)
        
        save_grid_image(samples, save_path, scale=scale, normalize=normalize)
    
    return samples


def save_vae_recon_grid(model, configs, dataloader, device, epoch=None, train=False, scale=4):    
    with torch.no_grad():
        data_iter = iter(dataloader)
        images, _ = next(data_iter)
        images = images.to(device)
        
        x = images
        x_hat, _, _ = model(images)
        
        if configs["model"]['activation'] == 'tanh':
            x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
            x_hat = (x_hat + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
        x = x.clamp(0, 1).cpu()
        x_hat = x_hat.clamp(0, 1).cpu()
        
        fig_scale = max(1, scale)
        _, axes = plt.subplots(2, 8, figsize=(12 * fig_scale, 4 * fig_scale))
        for i in range(8):
            if images.shape[1] == 3:
                axes[0, i].imshow(x[i].permute(1, 2, 0))
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat[i].permute(1, 2, 0))
                axes[1, i].axis('off')
                axes[0, i].title.set_text("Original")
                axes[1, i].title.set_text("Reconstruction")
            else:        
                axes[0, i].imshow(x[i].permute(1, 2, 0).squeeze(), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(x_hat[i].permute(1, 2, 0).squeeze(), cmap='gray')
                axes[1, i].axis('off')
                axes[0, i].title.set_text("Original")
                axes[1, i].title.set_text("Reconstruction")
        plt.tight_layout()
        
        output_dir = get_output_dir(configs)
        
        if epoch is None:
            plt.savefig(os.path.join(output_dir, "visualization", f"reconstructions_final.png"))
        else:
            plt.savefig(os.path.join(output_dir, "visualization", "train" if train else "test", f"reconstructions_epoch_{epoch}.png"))
        
        plt.close()


def save_diffusion_sampling_gif(model, diffusion, configs, num_samples=16, capture_interval=20, scale=4, use_cfg=False, class_id=None, guidance_scale=None, save_dir=None, filename="sampling_process.gif", final_name=None, duration=200, sampling_steps=None):
    """Generate GIF of diffusion sampling process.
    
    Args:
        guidance_scale: If None, reads from config (for training). If specified, uses that value (for inference).
        save_dir: If None, uses config output_dir (for training). If specified, uses that path (for inference).
        sampling_steps: If None, reads from config or uses default. If specified, uses that value.
    """
    model.eval()
    device = diffusion.betas.device
    
    # Initialize recorder with appropriate save directory
    if save_dir is not None:
        recorder = SampleRecorder(configs, device=device, save_filename=filename, save_dir=save_dir, scale=scale)
    else:
        recorder = SampleRecorder(configs, device=device, scale=scale)
    
    # 1. Initialize noise (x_T)
    img_size = configs['model']['img_size']
    channels = configs['model']['in_channels']
    img = torch.randn((num_samples, channels, img_size, img_size), device=device)
    
    print("Sampling process visualization started...")
    
    with torch.no_grad():
        model_kwargs = None
        
        # Determine guidance_scale: use parameter if provided, else read from config
        if guidance_scale is None:
            guidance_scale = 1.0
            if use_cfg:
                guidance_scale = float(configs.get("diffusion", {}).get("guidance_scale", 1.0))
        
        if use_cfg:
            if class_id is None:
                num_classes = int(configs.get("dataset", {}).get("num_classes", 1000))
                class_id = int(torch.randint(0, num_classes, (1,), device=device).item())
            model_kwargs = {
                "y": torch.full((num_samples,), class_id, device=device, dtype=torch.long)
            }

        # [Reverse process loop] T -> 0
        if hasattr(diffusion, "_set_timesteps"):
            if sampling_steps is None:
                sampling_steps = int(configs.get("diffusion", {}).get("sampling_steps", 50))
            diffusion._set_timesteps(sampling_steps)
            total_steps = len(diffusion.timesteps)
            
            for i in range(total_steps):
                t_val = diffusion.timesteps[i]
                t = torch.full((num_samples,), t_val, device=device, dtype=torch.long)
                
                # 2. One denoising step (use p_sample)
                img = diffusion.p_sample(model, img, t, i, model_kwargs=model_kwargs, guidance_scale=guidance_scale)
                
                # 3. Capture frames at intervals (always include the final step)
                if i % capture_interval == 0 or i == total_steps - 1:
                    recorder.record_step(img, t=int(t_val.item()))
        else:
            total_steps = len(diffusion.betas)
            
            for i in reversed(range(total_steps)):
                t = torch.full((num_samples,), i, device=device, dtype=torch.long)
                
                # 2. One denoising step (use p_sample)
                img = diffusion.p_sample(model, img, t, model_kwargs=model_kwargs, guidance_scale=guidance_scale)
                
                # 3. Capture frames at intervals (always include the final step 0)
                if i % capture_interval == 0 or i == 0:
                    recorder.record_step(img, t=i)
                
    # 4. Save GIF
    recorder.save_gif(duration=duration)
    
    # 5. Optionally save final image as grid
    if final_name is not None and save_dir is not None:
        final_path = os.path.join(save_dir, final_name)
        save_grid_image(img, final_path, scale=scale, normalize=True)



