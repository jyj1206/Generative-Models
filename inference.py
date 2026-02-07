import os
import argparse
import yaml
import torch
from matplotlib import pyplot as plt

from models.build import build_model, build_diffusion_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--out_name', type=str, default='single_sample.png')
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


def save_sample(sample, save_path):
    img = sample[0].permute(1, 2, 0).squeeze()
    plt.figure(figsize=(3, 3))
    if img.ndim == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()
    configs = yaml_loader(args.configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    output_dir = os.path.dirname(model_dir)
    configs["output_dir"] = output_dir
    os.makedirs(os.path.join(output_dir, "inference"), exist_ok=True)
    model = build_model(configs)

    load_checkpoint(model, args.model_path, device)
    model.to(device)
    
    torch.manual_seed(42)

    diffusion = None
    if configs["task"] == "diffusion":
        diffusion, _ = build_diffusion_scheduler(configs, device)

    sample = sample_one(model, configs, device, diffusion=diffusion)
    save_path = os.path.join(output_dir, "inference", args.out_name)
    save_sample(sample, save_path)
    print(f"Saved single sample: {save_path}")


if __name__ == "__main__":
    main()