import os
import shutil
import math
import copy
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.build import build_dataset
from models.build import build_model, build_optimizer, build_diffusion_scheduler
from utils.util_ema import ema_update
from utils.util_visualization import generate_and_save_samples, save_stable_diffusion_sampling_gif, save_loss_curve
from utils.util_save import save_diffusion_checkpoint
from utils.util_logger import setup_train_logger
from utils.util_paths import build_output_dir
from utils.util_ddp import set_visible_gpus, setup_runtime, cleanup_ddp, is_main


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


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def parse_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        if len(batch) == 1:
            return batch[0], None
    return batch, None


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

    config_dest = os.path.join(output_dir, os.path.basename(config_path))
    try:
        shutil.copyfile(config_path, config_dest)
    except Exception as e:
        print(f"Warning: Failed to copy config file: {e}")
    return output_dir


def build_dataloader(configs):
    batch_size = configs["train"]["batch_size"]
    num_workers = configs["train"]["num_workers"]

    train_dataset, _ = build_dataset(configs)
    sampler = None
    if configs.get("distributed", False):
        sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
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

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
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
            ema_model.load_state_dict(unwrap_model(model).state_dict())

    return start_epoch, iterations


def load_vae_checkpoint(vae, configs, device):
    checkpoint_path = configs["model"]["autoencoder"].get("checkpoint_path", None)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading VAE checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict):
            state_dict = None
            for key in ('vae_gan_state_dict', 'vqgan_state_dict'):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                    state_dict = checkpoint
                else:
                    raise KeyError("Checkpoint must contain 'vae_gan_state_dict' or 'vqgan_state_dict'.")
        else:
            state_dict = checkpoint

        vae.load_state_dict(state_dict)

        for param in vae.parameters():
            param.requires_grad = False
    else:
        if checkpoint_path is not None:
            print(f"WARNING: VAE checkpoint not found at: {checkpoint_path}. Using randomly initialized VAE.")
        else:
            print("WARNING: No checkpoint_path specified. Using randomly initialized VAE.")

    return vae


def main():
    args = parse_args()
    configs = yaml_loader(args.config)
    set_visible_gpus(configs)
    distributed, local_rank, device = setup_runtime()
    configs["distributed"] = distributed

    output_dir = prepare_output_dir(configs, args.config, args.resume)
    logger = setup_train_logger(output_dir)
    if is_main():
        print(f"Logging to: {logger.log_path}")

    train_loader = build_dataloader(configs)

    models = build_model(configs)
    vae = models["autoencoder"].to(device)
    model = models["denoiser"].to(device)
    text_encoder = models.get("text_encoder", None)
    if text_encoder is not None:
        text_encoder = text_encoder.to(device)

    use_text = (
        configs.get("task") == "latent_diffusion"
        and configs.get("conditioning", {}).get("type") == "text"
        and text_encoder is not None
    )

    cfg_cfg = configs.get("conditioning", {}).get("cfg", {})
    cfg_enabled = bool(cfg_cfg.get("enabled", False))
    p_uncond = float(cfg_cfg.get("p_uncond", 0.0))

    null_text = ""
    base_uncond_context = None

    if text_encoder is not None:
        text_encoder.eval()
        for param in text_encoder.parameters():
            param.requires_grad = False

    if use_text:
        null_text = configs.get("conditioning", {}).get("text", {}).get("null_text", "")
        with torch.no_grad():
            base_uncond_context = text_encoder.encode_text([null_text], device=device)

    ema_model, ema_decay = setup_ema_model(model, configs, device)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    vae_to_use = load_vae_checkpoint(vae, configs, device)
    vae_to_use.eval()

    scale_factor = configs["model"].get("autoencoder", {}).get("scale_factor", None)
    if scale_factor is None:
        if is_main():
            print("Computing latent scaling factor...")
            with torch.no_grad():
                sample_batch = next(iter(train_loader))[0][:min(64, len(train_loader.dataset))].to(device)
                sample_latent = vae_to_use.encode(sample_batch)
                if isinstance(sample_latent, (tuple, list)):
                    sample_latent = sample_latent[0]
                std_val = sample_latent.std().item()
                print(f"Sample latent shape: {sample_latent.shape}, std: {std_val:.4f}")
                scale_factor = 1.0 / std_val
            print(f"Latent scaling factor: {scale_factor:.4f}")
        else:
            scale_factor = 0.0

        if distributed:
            scale_tensor = torch.tensor([scale_factor], dtype=torch.float32, device=device)
            torch.distributed.broadcast(scale_tensor, src=0)
            scale_factor = scale_tensor.item()
    else:
        scale_factor = float(scale_factor)
        if is_main():
            print(f"Using config latent scaling factor: {scale_factor:.4f}")

    configs["model"]["autoencoder"]["scale_factor"] = scale_factor

    optimizer = build_optimizer(model, configs)
    diffusion, num_timesteps = build_diffusion_scheduler(configs, device)
    num_epochs = compute_num_epochs(train_loader, configs)
    ema_enabled = ema_model is not None

    train_loss_history = []
    start_epoch = 0
    iterations = 0
    avg_train_loss = None

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

        if distributed:
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", disable=not is_main())

        for batch in pbar:
            inputs, cond = parse_batch(batch)
            inputs = inputs.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                latent = vae_to_use.encode(inputs)
                if isinstance(latent, (tuple, list)):
                    latent = latent[0]
                latent = latent * scale_factor

            model_kwargs = None

            if use_text:
                if isinstance(cond, (tuple, list)):
                    cond = list(cond)
                elif isinstance(cond, str):
                    cond = [cond]
                elif cond is None:
                    raise ValueError("Text conditioning is enabled, but dataset did not return text captions.")
                else:
                    cond = [str(c) for c in cond]

                with torch.no_grad():
                    context = text_encoder.encode_text(cond, device=device)

                uncond_context = base_uncond_context.expand(context.size(0), -1, -1).contiguous()

                if cfg_enabled and p_uncond > 0.0:
                    drop_mask = torch.rand(context.size(0), device=device) < p_uncond
                    if drop_mask.any():
                        context = context.clone()
                        context[drop_mask] = uncond_context[drop_mask]

                model_kwargs = {
                    "context": context,
                }

            t = torch.randint(0, num_timesteps, (latent.size(0),), device=device).long()

            loss = diffusion.p_losses(model, latent, t, model_kwargs=model_kwargs)
            loss.backward()
            optimizer.step()

            if ema_enabled:
                ema_update(unwrap_model(model), ema_model, ema_decay)

            train_loss_sum += loss.item()
            iterations += 1

            if is_main():
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.6f}",
                        "iter": iterations,
                    }
                )

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_loss_tensor = torch.tensor(avg_train_loss, device=device)

        if distributed:
            torch.distributed.all_reduce(avg_train_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_train_loss_tensor /= torch.distributed.get_world_size()

        avg_train_loss = avg_train_loss_tensor.item()
        train_loss_history.append(avg_train_loss)

        if is_main():
            print(f"Epoch [{epoch}/{num_epochs}] Done. | Train Loss: {avg_train_loss:.6f}")

            if epoch % 25 == 0:
                denoiser_to_use = ema_model if ema_enabled else unwrap_model(model)

                reference_model = {
                    "autoencoder": vae_to_use,
                    "denoiser": denoiser_to_use,
                }

                mid_model_kwargs = None
                if use_text:
                    reference_model["text_encoder"] = text_encoder
                    reference_model["text_encoder"].eval()
                    sample_batch = next(iter(train_loader))
                    _, sample_captions = parse_batch(sample_batch)
                    if isinstance(sample_captions, (tuple, list)):
                        sample_captions = list(sample_captions)[:16]
                    elif isinstance(sample_captions, str):
                        sample_captions = [sample_captions]
                    else:
                        sample_captions = [str(c) for c in sample_captions][:16]
                    while len(sample_captions) < 16:
                        sample_captions.append(sample_captions[-1])
                    with torch.no_grad():
                        mid_context = text_encoder.encode_text(sample_captions, device=device)
                        mid_uncond = base_uncond_context.expand(16, -1, -1).contiguous()
                    mid_model_kwargs = {"context": mid_context, "uncond_context": mid_uncond}

                reference_model["autoencoder"].eval()
                reference_model["denoiser"].eval()

                generate_and_save_samples(
                    reference_model,
                    configs,
                    device,
                    diffusion=diffusion,
                    num_samples=16,
                    epoch=epoch,
                    scale=args.scale,
                    model_kwargs=mid_model_kwargs,
                )

                save_diffusion_checkpoint(
                    unwrap_model(model),
                    optimizer,
                    avg_train_loss,
                    configs,
                    epoch,
                    iterations,
                    final=False,
                    ema_model=ema_model if ema_enabled else None,
                )

    if is_main():
        denoiser_to_use = ema_model if ema_enabled else unwrap_model(model)

        reference_model = {
            "autoencoder": vae_to_use,
            "denoiser": denoiser_to_use,
        }

        sample_model_kwargs = None
        if use_text:
            reference_model["text_encoder"] = text_encoder
            reference_model["text_encoder"].eval()
            sample_batch = next(iter(train_loader))
            _, sample_captions = parse_batch(sample_batch)
            if isinstance(sample_captions, (tuple, list)):
                sample_captions = list(sample_captions)[:16]
            elif isinstance(sample_captions, str):
                sample_captions = [sample_captions]
            else:
                sample_captions = [str(c) for c in sample_captions][:16]
            while len(sample_captions) < 16:
                sample_captions.append(sample_captions[-1])
            with torch.no_grad():
                sample_context = text_encoder.encode_text(sample_captions, device=device)
                sample_uncond = base_uncond_context.expand(16, -1, -1).contiguous()
            sample_model_kwargs = {"context": sample_context, "uncond_context": sample_uncond}

        reference_model["autoencoder"].eval()
        reference_model["denoiser"].eval()

        generate_and_save_samples(
            reference_model,
            configs,
            device,
            diffusion=diffusion,
            num_samples=16,
            scale=args.scale,
            model_kwargs=sample_model_kwargs,
        )
        save_stable_diffusion_sampling_gif(
            reference_model,
            diffusion,
            configs,
            num_samples=16,
            capture_interval=20,
            scale=args.scale,
            model_kwargs=sample_model_kwargs,
        )
        save_loss_curve(configs, train_loss_history)

        final_loss = avg_train_loss if avg_train_loss is not None else float("nan")
        save_diffusion_checkpoint(
            unwrap_model(model),
            optimizer,
            final_loss,
            configs,
            num_epochs,
            iterations,
            final=True,
            ema_model=ema_model if ema_enabled else None,
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()