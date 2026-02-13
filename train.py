import torch
import os
import argparse
import yaml
import copy
import math

from models.build import build_model, build_loss_function, build_optimizer, build_diffusion_scheduler
from datasets.build import build_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.util_makegif import TrainRecorder
from utils.util_visualization import generate_and_save_samples, save_loss_curve
from utils.util_visualization import save_vae_recon_grid, save_diffusion_sampling_gif
from utils.util_save import save_vae_checkpoint, save_gan_checkpoint, save_diffusion_checkpoint
from utils.util_ema import ema_update
from utils.util_paths import build_output_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--resume_ema', type=str, default=None)
    args = parser.parse_args()
    return args


def yaml_loader(configs_path):
    with open(configs_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs


def load_resume_state(task, model, optimizer, device, resume_path, optimizer_g=None, optimizer_d=None,
                      ema_model=None, resume_ema_path=None):
    checkpoint = torch.load(resume_path, map_location=device)
    start_epoch = int(checkpoint.get('epoch', 0))
    iterations = int(checkpoint.get('iterations', 0))

    if task == 'vae':
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif task == 'gan':
        model.netG.load_state_dict(checkpoint['generator_state_dict'])
        model.netD.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    elif task == 'diffusion':
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        raise ValueError(f"Unsupported task: {task}")

    if ema_model is not None:
        if resume_ema_path is None:
            candidate = resume_path.replace('.pth', '_ema.pth')
            if os.path.exists(candidate):
                resume_ema_path = candidate

        if resume_ema_path is not None and os.path.exists(resume_ema_path):
            ema_checkpoint = torch.load(resume_ema_path, map_location=device)
            ema_model.load_state_dict(ema_checkpoint['model_state_dict'])
        elif task == 'diffusion':
            # Fallback: initialize EMA from the resumed model weights
            ema_model.load_state_dict(model.state_dict())

    return start_epoch, iterations
    

def main():
    """
        Loading configurations
    """
    args = parse_args()
    configs = yaml_loader(args.config)
    
    task = configs["task"]

    run_name = os.path.splitext(os.path.basename(args.config))[0]
    configs["run_name"] = run_name
    if args.resume:
        model_dir = os.path.dirname(os.path.abspath(args.resume))
        output_dir = os.path.dirname(model_dir)
        configs["output_dir"] = output_dir
    else:
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
    
    test_dataloader = None
    
    if test_dataset is not None:
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
    steps_per_epoch = len(train_dataloader)
    if "iterations" in configs["train"]:
        iterations_target = int(configs["train"]["iterations"])
        num_epochs = max(1, math.ceil(iterations_target / steps_per_epoch))
    else:
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

    use_cfg = False
    p_uncond = 0.0
    num_classes = None
    if task != 'diffusion':
        criterion = build_loss_function(configs)
    else:
        diffusion, num_timesteps = build_diffusion_scheduler(configs, device)
        use_cfg = configs.get("diffusion", {}).get("use_cfg", False)  # Explicit CFG flag
        if use_cfg:
            num_classes = configs.get("dataset", {}).get("num_classes", None)
            if num_classes is None:
                raise ValueError("use_cfg=True requires dataset.num_classes to be set in config")
            p_uncond = float(configs.get("diffusion", {}).get("p_uncond", 0.0))
    
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

    start_epoch = 0
    iterations = 0

    if args.resume:
        start_epoch, iterations = load_resume_state(
            task,
            model,
            optimizer if task in ['vae', 'diffusion'] else None,
            device,
            args.resume,
            optimizer_g=optimizer_G if task == 'gan' else None,
            optimizer_d=optimizer_D if task == 'gan' else None,
            ema_model=ema_model if ema else None,
            resume_ema_path=args.resume_ema
        )
    
    for epoch in range(start_epoch + 1, num_epochs + 1):
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
                generate_and_save_samples(model.decoder, configs, device, num_samples=16, epoch=epoch, scale=args.scale)
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
                generate_and_save_samples(model.netG, configs, device, num_samples=16, epoch=epoch, scale=args.scale)
                save_gan_checkpoint(model, optimizer_G, optimizer_D, avg_loss_G, avg_loss_D, configs, epoch, iterations)
 
        elif task == 'diffusion':
            train_loss_sum = 0.0
            
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                model_kwargs = None
                
                if use_cfg: # for class-conditional diffusion
                    labels = labels.to(device)
                    if p_uncond > 0.0:
                        drop_mask = torch.rand(labels.shape, device=device) < p_uncond
                        labels = labels.clone()
                        labels[drop_mask] = diffusion.null_token_idx
                    model_kwargs = {"y": labels}
                
                optimizer.zero_grad()

                t = torch.randint(0, num_timesteps, (inputs.size(0),), device=device).long()
                
                loss = diffusion.p_losses(model, inputs, t, model_kwargs=model_kwargs)
                
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
                generate_and_save_samples(
                    ema_model if ema else model,
                    configs,
                    device,
                    diffusion=diffusion,
                    num_samples=16,
                    epoch=epoch,
                    scale=args.scale,
                    use_cfg=use_cfg
                )
                save_diffusion_checkpoint(model, optimizer, avg_train_loss, configs, epoch, iterations, final=False, ema_model=ema_model if configs['train']['ema'] else None)
            

    if task in ['vae', 'gan']:
        recorder.save_gif()
    
    model.eval()
    if task == 'vae':
        generate_and_save_samples(model.decoder, configs, device, num_samples=16, scale=args.scale)
        
        save_vae_recon_grid(model, configs, test_dataloader, device, scale=args.scale)
        
        save_loss_curve(configs, train_loss_history, test_loss_history)
        
        save_vae_checkpoint(model, optimizer, avg_train_loss, configs, num_epochs, iterations, final=True)
    
    elif task == 'gan':
        generate_and_save_samples(model.netG, configs, device, num_samples=16, scale=args.scale)
        
        save_loss_curve(configs, loss_G_history, loss_D_history, "Generator Loss", "Discriminator Loss")
        save_gan_checkpoint(model, optimizer_G, optimizer_D, avg_loss_G, avg_loss_D, configs, num_epochs, iterations, final=True)
    
    elif task == 'diffusion':
        generate_and_save_samples(
            ema_model if ema else model,
            configs,
            device,
            diffusion=diffusion,
            num_samples=16,
            scale=args.scale,
            use_cfg=use_cfg
        )
        save_diffusion_sampling_gif(
            ema_model if ema else model,
            diffusion,
            configs,
            num_samples=16,
            capture_interval=20,
            scale=args.scale,
            use_cfg=use_cfg
        )
        save_loss_curve(configs, train_loss_history)
        save_diffusion_checkpoint(model, optimizer, avg_train_loss, configs, num_epochs, iterations, final=True, ema_model=ema_model if ema else None)
    
if __name__ == "__main__":
    main()
    