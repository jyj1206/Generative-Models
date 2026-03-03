import os
import math
import copy
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.build import build_dataset
from models.build import build_model, build_optimizer, build_diffusion_scheduler
from utils.util_ema import ema_update
from utils.util_visualization import generate_and_save_samples, save_stable_diffusion_sampling_gif, save_loss_curve
from utils.util_save import save_diffusion_checkpoint
from utils.util_logger import setup_train_logger
from utils.util_paths import build_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model (DDPM/DDIM).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_ema", type=str, default=None)
    return parser.parse_args()


def yaml_loader(configs_path):
    with open(configs_path, "r") as f:
        return yaml.safe_load(f)


def prepare_output_dir(configs, config_path, resume_path):
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    configs["run_name"] = run_name

    if resume_path:
        model_dir = os.path.dirname(os.path.abspath(resume_path))
        output_dir = os.path.dirname(model_dir)
    else:
        output_dir = build_output_dir(configs, run_name)
    configs["output_dir"] = output_dir

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualization", "train"), exist_ok=True)
    return output_dir


def build_dataloader(configs):
    batch_size = configs["train"]["batch_size"]
    num_workers = configs["train"]["num_workers"]

    train_dataset, _ = build_dataset(configs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader


def compute_num_epochs(train_loader, configs):
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Train dataloader returned zero batches. Check dataset setup.")

    train_cfg = configs["train"]
    if "iterations" in train_cfg:
        iterations_target = int(train_cfg["iterations"])
        return max(1, math.ceil(iterations_target / steps_per_epoch))
    return int(train_cfg["epochs"])


def setup_ema_model(model, configs, device):
    if not configs["train"].get("ema", False):
        return None, None

    ema_model = copy.deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_decay = float(configs["train"]["ema_decay"])
    return ema_model, ema_decay


def load_resume_state(model, optimizer, device, resume_path, ema_model=None, resume_ema_path=None):
    checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0))
    iterations = int(checkpoint.get("iterations", 0))

    if ema_model is not None:
        if resume_ema_path is None:
            candidate = resume_path.replace(".pth", "_ema.pth")
            if os.path.exists(candidate):
                resume_ema_path = candidate
        if resume_ema_path and os.path.exists(resume_ema_path):
            ema_ckpt = torch.load(resume_ema_path, map_location=device, weights_only=True)
            ema_model.load_state_dict(ema_ckpt["model_state_dict"])
        else:
            ema_model.load_state_dict(model.state_dict())

    return start_epoch, iterations


def load_vae_checkpoint(vae, configs, device):
    checkpoint_path = configs["first_stage"].get("checkpoint_path", None)
    
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading VAE checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        if isinstance(checkpoint, dict):
            if "vqvae_state_dict" in checkpoint:
                state_dict = checkpoint["vqvae_state_dict"]
            elif all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                state_dict = checkpoint
            else:
                raise KeyError("Checkpoint must contain 'vqvae_state_dict")
        else:
            state_dict = checkpoint

        vae.load_state_dict(state_dict)
       
        for param in vae.parameters():
            param.requires_grad = False
    else:
        raise ValueError(f"VAE checkpoint not found at: {checkpoint_path}")

    return vae


def main():
    args = parse_args()
    configs = yaml_loader(args.config)

    output_dir = prepare_output_dir(configs, args.config, args.resume)
    logger = setup_train_logger(output_dir)
    print(f"Logging to: {logger.log_path}")
    train_loader = build_dataloader(configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = build_model(configs)
    
    vae = models['first_stage'].to(device)
    model = models['second_stage'].to(device)
    
    # Load VAE
    vae = load_vae_checkpoint(vae, configs, device)
    vae.eval()
    
    optimizer = build_optimizer(model, configs)   
    
    diffusion, num_timesteps = build_diffusion_scheduler(configs, device)
    num_epochs = compute_num_epochs(train_loader, configs)
    
    ema_model, ema_decay = setup_ema_model(model, configs, device)
    ema_enabled = ema_model is not None

    train_loss_history = []
    start_epoch = 0
    iterations = 0

    if args.resume:
        start_epoch, iterations = load_resume_state(
            model,
            optimizer,
            device,
            args.resume,
            ema_model=ema_model,
            resume_ema_path=args.resume_ema,
        )
        
    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]")

        for inputs, _ in pbar:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            
            with torch.no_grad():
                z = vae.encode(inputs)
            
            t = torch.randint(0, num_timesteps, (inputs.size(0),), device=device).long()
            loss = diffusion.p_losses(model, z, t)
            loss.backward()
            optimizer.step()

            if ema_enabled:
                ema_update(model, ema_model, ema_decay)

            train_loss_sum += loss.item()
            iterations += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iter": iterations})

        avg_train_loss = train_loss_sum / len(train_loader)
        train_loss_history.append(avg_train_loss)
        print(f"Epoch [{epoch}/{num_epochs}] Done. | Train Loss: {avg_train_loss:.4f}")

        if epoch % 2 == 0:
            reference_model = {
                "first_stage": vae,
                "second_stage": ema_model if ema_enabled else model,
            }
            reference_model["first_stage"].eval()
            reference_model["second_stage"].eval()
            generate_and_save_samples(reference_model, configs, device, diffusion=diffusion, num_samples=16, epoch=epoch, scale=args.scale,)
            save_diffusion_checkpoint(model, optimizer, avg_train_loss, configs, epoch, iterations, final=False, ema_model=ema_model if ema_enabled else None)

    reference_model = {
        "first_stage": vae,
        "second_stage": ema_model if ema_enabled else model,
    }
    reference_model["first_stage"].eval()
    reference_model["second_stage"].eval()
    generate_and_save_samples(reference_model, configs, device, diffusion=diffusion, num_samples=16, scale=args.scale)
    save_stable_diffusion_sampling_gif(reference_model, diffusion, configs, num_samples=16, capture_interval=20, scale=args.scale)
    save_loss_curve(configs, train_loss_history)
    save_diffusion_checkpoint(model, optimizer, avg_train_loss, configs, num_epochs, iterations, final=True, ema_model=ema_model if ema_enabled else None)


if __name__ == "__main__":
    main()    

  

