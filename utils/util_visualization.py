import os
import torch
from matplotlib import pyplot as plt


def visualize_loss_curve(configs, train_loss_history, test_loss_history):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(test_loss_history, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join("output", configs["task_name"], "visualization", "loss_curve.png"))
    plt.close()


def visualize_generated_samples(model, configs, device, epoch=None):    
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
            plt.savefig(os.path.join("output", configs["task_name"], "visualization", f"reconstructions_final.png"))
        else:
            plt.savefig(os.path.join("output", configs["task_name"], "visualization", "train", f"generated_samples_epoch_{epoch}.png" if epoch else "final_generated_samples.png"))
        
        plt.close() 


def visualize_reconstructions(model, configs, dataloader, device, epoch=None, train=False):    
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
            plt.savefig(os.path.join("output", configs["task_name"], "visualization", f"reconstructions_final.png"))
        else:
            plt.savefig(os.path.join("output", configs["task_name"], "visualization", "train" if train else "test", f"reconstructions_epoch_{epoch}.png"))
        
        plt.close()
        
