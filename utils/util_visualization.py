import os
import torch
from matplotlib import pyplot as plt
from utils.util_makegif import SampleRecorder
from utils.util_paths import get_output_dir


def plot_loss(configs, loss_history1, loss_history2=None,
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


def plot_samples(model, configs, device, epoch=None):    
    with torch.no_grad():
        z = torch.randn(16, model.latent_dim).to(device)
        samples = model(z)
        
        if configs["model"]['activation'] == 'tanh':
            samples = (samples + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        samples = samples.clamp(0, 1).cpu()
        
        _, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            img = samples[i].permute(1, 2, 0).squeeze()
            if img.shape[-1] == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        
        if epoch is None:
            output_dir = get_output_dir(configs)
            plt.savefig(os.path.join(output_dir, "visualization", f"reconstructions_final.png"))
        else:
            output_dir = get_output_dir(configs)
            plt.savefig(os.path.join(output_dir, "visualization", "train", f"generated_samples_epoch_{epoch}.png" if epoch else "final_generated_samples.png"))
        
        plt.close() 


def plot_vae_recon(model, configs, dataloader, device, epoch=None, train=False):    
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
        
        _, axes = plt.subplots(2, 8, figsize=(12, 4))
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
        
        if epoch is None:
            output_dir = get_output_dir(configs)
            plt.savefig(os.path.join(output_dir, "visualization", f"reconstructions_final.png"))
        else:
            output_dir = get_output_dir(configs)
            plt.savefig(os.path.join(output_dir, "visualization", "train" if train else "test", f"reconstructions_epoch_{epoch}.png"))
        
        plt.close()
        

def plot_diffusion_samples(model, configs, diffusion):    
    with torch.no_grad():
        sample_shape = (16, configs["model"]['in_channels'], configs["model"]['img_size'], configs["model"]['img_size'])
        samples = diffusion.p_sample_loop(model, shape=sample_shape)
        
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1).cpu()
        
        _, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            img = samples[i].permute(1, 2, 0).squeeze()
            if img.shape[-1] == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        
        output_dir = get_output_dir(configs)
        plt.savefig(os.path.join(output_dir, "visualization", f"diffusion_generated_samples.png"))
        
        plt.close()
        

def run_sampling_plot(model, diffusion, configs, num_samples=16, capture_interval=20):
    model.eval()
    device = diffusion.betas.device
    recorder = SampleRecorder(configs, device='cuda')
    
    # 1. Initialize noise (x_T)
    img_size = recorder.configs['model']['img_size']
    channels = recorder.configs['model']['in_channels']
    img = torch.randn((num_samples, channels, img_size, img_size), device=device)
    
    print("Sampling process visualization started...")
    
    with torch.no_grad():
        # [Reverse process loop] T -> 0
        # diffusion.betas length is usually 1000
        total_steps = len(diffusion.betas)
        
        for i in reversed(range(total_steps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            
            # 2. One denoising step (use p_sample)
            img = diffusion.p_sample(model, img, t)
            
            # 3. Capture frames at intervals (always include the final step 0)
            if i % capture_interval == 0 or i == 0:
                recorder.record_step(img, t=i)
                
    # 4. Save GIF
    recorder.save_gif(duration=200)  # Adjust duration to control playback speed

