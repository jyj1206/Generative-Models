import os
import torch    

def save_vae_model_checkpoint(model, optimizer, loss, configs, epoch, final=False):    
    if final:
        checkpoint_path = os.path.join("output", configs["task_name"], "checkpoints", "model_final.pth")
    else:
        checkpoint_path = os.path.join("output", configs["task_name"], "checkpoints", f"model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
def save_gan_model_checkpoint(model, optimizer_g, optimzer_d, loss_g, loss_d, configs, epoch, final=False):
    if final:
        checkpoint_path = os.path.join("output", configs["task_name"], "checkpoints", "gan_model_final.pth")
    else:
        checkpoint_path = os.path.join("output", configs["task_name"], "checkpoints", f"gan_model_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'generator_state_dict': model.netG.state_dict(),
        'discriminator_state_dict': model.netD.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimzer_d.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d
    }, checkpoint_path)