import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import argparse
import yaml
import torch

from torch.utils.data import DataLoader

from datasets.build import build_dataset
from models.build import build_model, build_diffusion_scheduler
from utils.util_paths import append_timestamp, get_timestamp
from utils.util_visualization import save_diffusion_sampling_gif, generate_and_save_samples, save_stable_diffusion_sampling_gif, save_vae_recon_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--out_name', type=str, default='single_sample.png')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--diffusion_gif', action='store_true')
    parser.add_argument('--grid_samples', type=int, default=1,
                        help="If 1, save a single image. If >1, save one grid image with that many samples.")
    parser.add_argument('--sampling_steps', type=int, default=None)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--class_id', type=int, default=None)
    parser.add_argument('--gif_interval', type=int, default=10)
    parser.add_argument('--gif_duration', type=int, default=50)
    parser.add_argument('--gif_name', type=str, default='sampling_process.gif')
    parser.add_argument('--gif_final_name', type=str, default='sampling_final.png')
    parser.add_argument('--condition_text', type=str, default=None, help='Text for conditional generation (latent_diffusion only)')
    parser.add_argument('--uncondition_text', type=str, default=None, help='Text for unconditional generation (latent_diffusion only)')
    args = parser.parse_args()
    return args


def yaml_loader(configs_path):
    with open(configs_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "generator_state_dict" in checkpoint:
        model.netG.load_state_dict(checkpoint["generator_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def load_vae_checkpoint(vae, checkpoint_path, device):
    """Load VAE checkpoint for stable diffusion inference."""
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading VAE checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict):
            if "vqvae_state_dict" in checkpoint:
                state_dict = checkpoint["vqvae_state_dict"]
            elif all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
                state_dict = checkpoint
            else:
                raise KeyError("Checkpoint must contain 'vqvae_state_dict'")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    output_dir = os.path.dirname(model_dir)
    configs["output_dir"] = output_dir
    inference_dir = os.path.join(output_dir, "inference")

    task = configs["task"]

    if task in ("diffusion", "latent_diffusion"):
        scheduler = configs.get("diffusion", {}).get("scheduler")
        if scheduler:
            inference_dir = os.path.join(inference_dir, scheduler)
    os.makedirs(inference_dir, exist_ok=True)

    model = build_model(configs)

    # --- Load checkpoints based on task ---
    if task == "vae" and configs["model"]["type"] == "vqvae":
        vqvae = model['vqvae'].to(device)
        load_vae_checkpoint(vqvae, args.model_path, device)
        vqvae.eval()
        generator = vqvae

    elif task == "latent_diffusion":
        vae = model['autoencoder'].to(device)
        unet = model['denoiser'].to(device)
        text_encoder = model.get('text_encoder', None)

        # Load VAE checkpoint from config
        vae_path = configs["model"].get("autoencoder", {}).get("checkpoint_path")
        vae = load_vae_checkpoint(vae, vae_path, device)
        vae.eval()

        # Load UNet checkpoint
        load_checkpoint(unet, args.model_path, device)
        unet.eval()

        generator = {'autoencoder': vae, 'denoiser': unet}
        if text_encoder is not None:
            text_encoder = text_encoder.to(device)
            text_encoder.eval()
            generator['text_encoder'] = text_encoder
    else:
        load_checkpoint(model, args.model_path, device)
        model.to(device)
        model.eval()

        if task == "vae":
            generator = model.decoder
        elif task == "gan":
            generator = model.netG
        else:
            generator = model

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # --- VQVAE reconstruction inference (early return) ---
    if task == "vae" and configs["model"]["type"] == "vqvae":
        timestamp = get_timestamp()
        start_time = time.perf_counter()

        num_recon = args.grid_samples if args.grid_samples > 1 else 16
        train_dataset, _ = build_dataset(configs)
        recon_loader = DataLoader(train_dataset, batch_size=num_recon, shuffle=True, num_workers=0)

        save_filename = append_timestamp("vqvae_reconstruction.png", timestamp)
        save_vae_recon_grid(generator, configs, recon_loader, device, scale=args.scale, num_samples=num_recon)

        elapsed = time.perf_counter() - start_time
        print(f"Saved VQVAE reconstruction to: {os.path.join(configs['output_dir'], 'visualization')}")
        print(f"Inference time: {elapsed:.3f} sec")
        return

    # --- Build diffusion scheduler if needed ---
    diffusion = None
    use_cfg = False
    if task in ("diffusion", "latent_diffusion"):
        diffusion, _ = build_diffusion_scheduler(configs, device)

        if task == "diffusion":
            cfg_cfg = configs.get("conditioning", {}).get("cfg", {})
            use_cfg = cfg_cfg.get("enabled", False)

            # Validate class_id if provided
            if args.class_id is not None:
                if not use_cfg:
                    raise ValueError("class_id was provided but use_cfg=False in config.")
                num_classes = configs.get("conditioning", {}).get("class", {}).get("num_classes", None)
                if num_classes is None:
                    raise ValueError("class_id requires conditioning.class.num_classes to be set in config.")
                if args.class_id < 0 or args.class_id >= int(num_classes):
                    raise ValueError(f"class_id must be in [0, {int(num_classes) - 1}].")

    timestamp = get_timestamp()
    start_time = time.perf_counter()

    grid_requested = args.grid_samples and args.grid_samples > 1
    num_to_sample = args.grid_samples if grid_requested else 1
    _, out_ext = os.path.splitext(args.out_name)
    out_ext = out_ext if out_ext else ".png"
    if grid_requested:
        save_filename = append_timestamp(f"gridx{num_to_sample}_sample{out_ext}", timestamp)
        save_path = os.path.join(inference_dir, save_filename)
    else:
        save_filename = append_timestamp(f"single_sample{out_ext}", timestamp)
        save_path = os.path.join(inference_dir, save_filename)

    # latent_diffusion text condition
    model_kwargs = None
    if task == "latent_diffusion":
        cond_type = configs.get("conditioning", {}).get("type", None)
        if cond_type != "text":
            if args.condition_text or args.uncondition_text:
                raise ValueError("condition_text/uncondition_text provided, but model is not a text-conditioned latent_diffusion model.")
        text_encoder = generator.get('text_encoder', None)
        if args.condition_text is not None:
            if text_encoder is None:
                raise ValueError("Text encoder not found in model for latent_diffusion.")
            batch_size = num_to_sample
            with torch.no_grad():
                # CLIP/BERT 등에서 encode_text는 batch 입력을 받음
                context = text_encoder.encode_text([args.condition_text]*batch_size, device=device)
                # context shape: (batch, dim) or (batch, seq_len, dim)
                if args.uncondition_text is not None:
                    uncond_context = text_encoder.encode_text([args.uncondition_text]*batch_size, device=device)
                else:
                    null_text = configs.get("conditioning", {}).get("text", {}).get("null_text", "")
                    uncond_context = text_encoder.encode_text([null_text]*batch_size, device=device)
                # shape, dtype, device 맞춤
                context = context.to(device)
                uncond_context = uncond_context.to(device)
            model_kwargs = {"context": context, "uncond_context": uncond_context}

    generate_and_save_samples(
        generator,
        configs,
        device,
        diffusion=diffusion if task in ("diffusion", "latent_diffusion") else None,
        sampling_steps=args.sampling_steps,
        use_cfg=use_cfg,
        class_id=args.class_id,
        guidance_scale=args.guidance_scale,
        num_samples=num_to_sample,
        save_to_file=True,
        save_dir=inference_dir,
        filename=save_filename,
        scale=args.scale,
        model_kwargs=model_kwargs,
    )

    elapsed = time.perf_counter() - start_time
    if grid_requested:
        print(f"Saved grid sample: {save_path}")
    else:
        print(f"Saved single sample: {save_path}")

    print(f"Inference time: {elapsed:.3f} sec")

    if task == "diffusion" and args.diffusion_gif:
        gif_name = append_timestamp(args.gif_name, timestamp)
        final_name = append_timestamp(args.gif_final_name, timestamp)
        save_diffusion_sampling_gif(
            model,
            diffusion,
            configs,
            num_samples=16,
            capture_interval=args.gif_interval,
            scale=args.scale,
            use_cfg=use_cfg,
            class_id=args.class_id,
            guidance_scale=args.guidance_scale,
            save_dir=inference_dir,
            filename=gif_name,
            final_name=final_name,
            duration=args.gif_duration,
            sampling_steps=args.sampling_steps
        )
    elif task == "latent_diffusion" and args.diffusion_gif:
        gif_name = append_timestamp(args.gif_name, timestamp)
        final_name = append_timestamp(args.gif_final_name, timestamp)

        gif_model_kwargs = None
        gif_text_encoder = generator.get('text_encoder', None)
        if gif_text_encoder is not None:
            gif_num_samples = 16
            null_text = configs.get("conditioning", {}).get("text", {}).get("null_text", "")
            with torch.no_grad():
                if args.condition_text is not None:
                    gif_context = gif_text_encoder.encode_text([args.condition_text] * gif_num_samples, device=device)
                else:
                    gif_context = gif_text_encoder.encode_text([null_text] * gif_num_samples, device=device)
                if args.uncondition_text is not None:
                    gif_uncond = gif_text_encoder.encode_text([args.uncondition_text] * gif_num_samples, device=device)
                else:
                    gif_uncond = gif_text_encoder.encode_text([null_text] * gif_num_samples, device=device)
            gif_model_kwargs = {"context": gif_context, "uncond_context": gif_uncond}

        save_stable_diffusion_sampling_gif(
            generator,
            diffusion,
            configs,
            num_samples=16,
            capture_interval=args.gif_interval,
            scale=args.scale,
            save_dir=inference_dir,
            filename=gif_name,
            final_name=final_name,
            duration=args.gif_duration,
            sampling_steps=args.sampling_steps,
            model_kwargs=gif_model_kwargs,
            guidance_scale=args.guidance_scale,
        )


if __name__ == "__main__":
    main()