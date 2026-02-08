import os
import torch
from utils.util_paths import get_output_dir

def save_vae_checkpoint(model, optimizer, loss, configs, epoch, iterations, final=False):    
    output_dir = get_output_dir(configs)
    if final:
        checkpoint_path = os.path.join(output_dir, "checkpoints", "model_final.pth")
    else:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    
def save_gan_checkpoint(model, optimizer_g, optimzer_d, loss_g, loss_d, configs, epoch, iterations, final=False):
    output_dir = get_output_dir(configs)
    if final:
        checkpoint_path = os.path.join(output_dir, "checkpoints", "gan_model_final.pth")
    else:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"gan_model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': model.netG.state_dict(),
        'discriminator_state_dict': model.netD.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimzer_d.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d
    }, checkpoint_path)
    
    
def save_diffusion_checkpoint(model, optimizer, loss, configs, epoch, iterations, final=False, ema_model= None):
    output_dir = get_output_dir(configs)
    if final:
        checkpoint_path = os.path.join(output_dir, "checkpoints", "diffusion_model_final.pth")
    else:
        checkpoint_path = os.path.join(output_dir, "checkpoints", f"diffusion_model_epoch_{epoch}.pth")
        
    if ema_model is not None:
        ema_checkpoint_path = checkpoint_path.replace(".pth", "_ema.pth")
        torch.save({
            'epoch': epoch,
            'iterations': iterations,
            'model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, ema_checkpoint_path)
    
    torch.save({
        'epoch': epoch,
        'iterations': iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    