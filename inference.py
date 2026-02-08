import os
import argparse
import yaml
import torch

from models.build import build_model, build_diffusion_scheduler
from utils.util_makegif import SampleRecorder
from utils.util_paths import append_timestamp, get_timestamp
from utils.util_visualization import save_grid_image, save_single_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--out_name', type=str, default='single_sample.png')
    parser.add_argument('--scale', type=int, default=4)
    rng_group = parser.add_mutually_exclusive_group()
    rng_group.add_argument('--random', action='store_true')
    rng_group.add_argument('--seed', type=int)
    parser.add_argument('--diffusion_gif', action='store_true')
    parser.add_argument('--gif_interval', type=int, default=10)
    parser.add_argument('--gif_duration', type=int, default=50)
    parser.add_argument('--gif_name', type=str, default='sampling_process.gif')
    parser.add_argument('--gif_final_name', type=str, default='sampling_final.png')
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


def sample_one(model, configs, device, diffusion=None):
    task = configs["task"]
    model.eval()
    with torch.no_grad():
        if task == "vae":
            z = torch.randn(1, model.latent_dim, device=device)
            sample = model.decoder(z)
            if configs["model"].get("activation") == "tanh":
                sample = (sample + 1) / 2
        elif task == "gan":
            z = torch.randn(1, model.netG.latent_dim, device=device)
            sample = model.netG(z)
            if configs["model"].get("activation") == "tanh":
                sample = (sample + 1) / 2
        elif task == "diffusion":
            if diffusion is None:
                raise ValueError("Diffusion sampler is required.")
            in_channels = int(configs["model"]["in_channels"])
            img_size = int(configs["model"]["img_size"])
            sample = diffusion.p_sample_loop(model, shape=(1, in_channels, img_size, img_size))
            sample = (sample + 1) / 2
        else:
            raise ValueError(f"Unsupported task: {task}")

    return sample.clamp(0, 1).cpu()


def save_diffusion_gif(model, diffusion, configs, save_dir, filename, final_name=None, capture_interval=200, duration=100, scale=4):
    device = diffusion.betas.device
    num_samples = 16
    img_size = int(configs["model"]["img_size"])
    channels = int(configs["model"]["in_channels"])

    recorder = SampleRecorder(configs, device=device, save_filename=filename, save_dir=save_dir, scale=scale)
    img = torch.randn((num_samples, channels, img_size, img_size), device=device)

    with torch.no_grad():
        total_steps = len(diffusion.betas)
        for i in reversed(range(total_steps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            img = diffusion.p_sample(model, img, t)
            if i % capture_interval == 0 or i == 0:
                recorder.record_step(img, t=i)

    recorder.save_gif(duration=duration)

    if final_name is not None:
        final_path = os.path.join(save_dir, final_name)
        save_grid_image(img, final_path, scale=scale, normalize=True)


def main():
    args = parse_args()
    configs = yaml_loader(args.configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    output_dir = os.path.dirname(model_dir)
    configs["output_dir"] = output_dir
    inference_dir = os.path.join(output_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    model = build_model(configs)

    load_checkpoint(model, args.model_path, device)
    model.to(device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    diffusion = None
    if configs["task"] == "diffusion":
        diffusion, _ = build_diffusion_scheduler(configs, device)

    timestamp = get_timestamp()
    sample = sample_one(model, configs, device, diffusion=diffusion)
    save_path = os.path.join(inference_dir, append_timestamp(args.out_name, timestamp))
    save_single_image(sample, save_path, scale=args.scale)
    print(f"Saved single sample: {save_path}")

    if configs["task"] == "diffusion" and args.diffusion_gif:
        gif_name = append_timestamp(args.gif_name, timestamp)
        final_name = append_timestamp(args.gif_final_name, timestamp)
        save_diffusion_gif(
            model,
            diffusion,
            configs,
            save_dir=inference_dir,
            filename=gif_name,
            final_name=final_name,
            capture_interval=args.gif_interval,
            duration=args.gif_duration,
            scale=args.scale
        )


if __name__ == "__main__":
    main()