import torch
import os
import argparse
import yaml
import copy

from models.build import build_model, build_loss_function, build_optimizer, build_diffusion_scheduler
from datasets.build import build_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.util_makegif import TrainRecorder
from utils.util_visualization import save_latent_samples_grid, save_loss_curve
from utils.util_visualization import save_vae_recon_grid, save_diffusion_samples_grid, save_diffusion_sampling_gif
from utils.util_save import save_vae_checkpoint, save_gan_checkpoint, save_diffusion_checkpoint
from utils.util_ema import ema_update
from utils.util_paths import build_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()
    return args


def yaml_loader(configs_path):
    with open(configs_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs
    

def main():
    """
        Loading configurations
    """
    args = parse_args()
    configs = yaml_loader(args.config)
    
    task = configs["task"]

    run_name = os.path.splitext(os.path.basename(args.config))[0]
    configs["run_name"] = run_name
    configs["output_dir"] = build_output_dir(configs, run_name)
    output_dir = configs["output_dir"]
    
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualization", "train"), exist_ok=True)

    if task == 'vae':
        os.makedirs(os.path.join(output_dir, "visualization", "test"), exist_ok=True)

    """
        Building dataset and dataloaders
    """
    batch_size = configs["train"]["batch_size"]
    num_workers = configs["train"]["num_workers"]
    
    train_dataset, test_dataset = build_dataset(configs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    """
        Building model
    """
    model = build_model(configs)
    
    """
        Training model
    """
    num_epochs = int(configs["train"]["epochs"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    if task == 'vae':
        optimizer = build_optimizer(model, configs)
    elif task == 'gan':
        optimizer_G = build_optimizer(model.netG, configs)
        optimizer_D = build_optimizer(model.netD, configs)
    elif task == 'diffusion':
        optimizer = build_optimizer(model, configs)
    else:
        raise ValueError(f"Unsupported task: {task}")

    if task != 'diffusion':
        criterion = build_loss_function(configs)
    else:
        diffusion, num_timesteps = build_diffusion_scheduler(configs, device)
    
    if task in ['vae', 'gan']:
        recorder = TrainRecorder(configs, device, num_samples=16, scale=args.scale)

    
    ema = configs['train'].get('ema', False)
    if ema:
        ema_decay = configs['train']['ema_decay']
        ema_model = copy.deepcopy(model).to(device)
        
        for param in ema_model.parameters():
            param.requires_grad = False

    
    train_loss_history = []
    test_loss_history = []
    
    loss_G_history = []
    loss_D_history = []

    iterations = 0
    
    for epoch in range(1, num_epochs + 1):
        model.train()

        pbar = tqdm(train_dataloader, desc=f"Epoch [{epoch}/{num_epochs}]")

        if task == 'vae':
            train_loss_sum = 0.0
            
            for inputs, _ in pbar:
                inputs = inputs.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item()

                iterations += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}, iter': {iterations}"})
            
            avg_train_loss = train_loss_sum / len(train_dataloader.dataset)
            train_loss_history.append(avg_train_loss)
        
            # -------------------------------------------------------
            # Test Loop (Validation)
            # -------------------------------------------------------
            model.eval()
            test_loss_sum = 0.0
            
            recorder.record_frame(model.decoder, epoch)
            
            with torch.no_grad():
                for test_inputs, _ in test_dataloader:
                    test_inputs = test_inputs.to(device)
                    test_outputs = model(test_inputs)
                    test_loss = criterion(test_outputs, test_inputs)
                    test_loss_sum += test_loss.item()
                    
            avg_test_loss = test_loss_sum / len(test_dataloader.dataset)
            test_loss_history.append(avg_test_loss)
            
            print(f"Epoch [{epoch}/{num_epochs}] Done. | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
            

            if epoch % 20 == 0:            
                save_latent_samples_grid(model.decoder, configs, device, epoch=epoch, scale=args.scale)
                save_vae_recon_grid(model, configs, train_dataloader, device, epoch, train=True, scale=args.scale)
                save_vae_recon_grid(model, configs, test_dataloader, device, epoch, train=False, scale=args.scale)
                
                save_vae_checkpoint(model, optimizer, avg_train_loss, configs, epoch, iterations)
 
        elif task == 'gan':
            loss_D_sum = 0.0
            loss_G_sum = 0.0

            for real_images, _ in pbar:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                z = torch.randn(batch_size, model.netG.latent_dim).to(device)
                fake_images = model.netG(z)
                label_real = torch.ones(batch_size).to(device) * 0.9
                label_fake = torch.zeros(batch_size).to(device)
                
                output_real = model.netD(real_images)
                output_fake = model.netD(fake_images.detach()) # Generator no update
                
                loss_D_real = criterion(output_real, label_real)
                loss_D_fake = criterion(output_fake, label_fake)
                
                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward()
                optimizer_D.step()
                
                # Train Generator
                optimizer_G.zero_grad()
                
                output = model.netD(fake_images)
                
                loss_G = criterion(output, label_real) # Trick: want fake to be real
                loss_G.backward()
                optimizer_G.step()

                loss_D_sum += loss_D.item()
                loss_G_sum += loss_G.item()
                
                iterations += 1
                
                pbar.set_postfix({'loss_D': f"{loss_D.item():.4f}", 'loss_G': f"{loss_G.item():.4f}", 'iter': f"{iterations}"})
                
            avg_loss_D = loss_D_sum / len(train_dataloader)
            avg_loss_G = loss_G_sum / len(train_dataloader)

            loss_D_history.append(avg_loss_D)
            loss_G_history.append(avg_loss_G)
            
            print(f"Epoch [{epoch}/{num_epochs}] Done. | Loss D: {avg_loss_D:.4f} | Loss G: {avg_loss_G:.4f}")
            recorder.record_frame(model.netG, epoch)
            
            if epoch % 25 == 0:  
                save_latent_samples_grid(model.netG, configs, device, epoch=epoch, scale=args.scale)
                save_gan_checkpoint(model, optimizer_G, optimizer_D, avg_loss_G, avg_loss_D, configs, epoch, iterations)
 
        elif task == 'diffusion':
            train_loss_sum = 0.0
            
            for inputs, _ in pbar:
                inputs = inputs.to(device)
                
                optimizer.zero_grad()

                t = torch.randint(0, num_timesteps, (inputs.size(0),), device=device).long()
                
                loss = diffusion.p_losses(model, inputs, t)
                
                loss.backward()
                optimizer.step()
                
                if configs['train']['ema']:
                    ema_update(model, ema_model, ema_decay)
                
                train_loss_sum += loss.item()
                
                iterations += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'iter': f"{iterations}"})
                
            avg_train_loss = train_loss_sum / len(train_dataloader)
            train_loss_history.append(avg_train_loss)
            print(f"Epoch [{epoch}/{num_epochs}] Done. | Train Loss: {avg_train_loss:.4f}")
            
            if epoch % 25 == 0:
                save_diffusion_samples_grid(ema_model if ema else model, configs, diffusion, epoch=epoch, scale=args.scale)
                save_diffusion_checkpoint(model, optimizer, avg_train_loss, configs, epoch, iterations, final=False, ema_model=ema_model if configs['train']['ema'] else None)
            

    if task in ['vae', 'gan']:
        recorder.save_gif()
    
    model.eval()
    if task == 'vae':
        save_latent_samples_grid(model.decoder, configs, device, scale=args.scale)
        
        save_vae_recon_grid(model, configs, test_dataloader, device, scale=args.scale)
        
        save_loss_curve(configs, train_loss_history, test_loss_history)
        save_vae_checkpoint(model, optimizer, avg_train_loss, configs, num_epochs, iterations, final=True)
    
    elif task == 'gan':
        save_latent_samples_grid(model.netG, configs, device, scale=args.scale)
        
        save_loss_curve(configs, loss_G_history, loss_D_history, "Generator Loss", "Discriminator Loss")
        save_gan_checkpoint(model, optimizer_G, optimizer_D, avg_loss_G, avg_loss_D, configs, num_epochs, iterations, final=True)
    
    elif task == 'diffusion':
        save_diffusion_samples_grid(ema_model if ema else model, configs, diffusion, scale=args.scale)
        save_diffusion_sampling_gif(
            ema_model if ema else model,
            diffusion,
            configs,
            num_samples=16,
            capture_interval=20,
            scale=args.scale
        )
        
        save_loss_curve(configs, train_loss_history)
        save_diffusion_checkpoint(model, optimizer, avg_train_loss, configs, num_epochs, iterations, final=True, ema_model=ema_model if ema else None)
    
    
if __name__ == "__main__":
    main()
    