import os
import torch
import torch.nn as nn
import argparse
import yaml
from models.build import build_model
from torchinfo import summary
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--configs', type=str, required=True)
    args = parser.parse_args()
    return args


def yaml_loader(configs_path):
    with open(configs_path, 'r') as f:
        configs = yaml.safe_load(f)
    return configs


def latent_sample(model, z_dim, num_samples, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim).to(device)
        samples = model.decoder(z)
    return samples


def visualize_samples(model, configs, device, num_samples=16):
    samples = latent_sample(model, model.latent_dim, num_samples, device)
    
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
    plt.savefig(os.path.join("output", configs["task"], "inference", "inference_generated_samples.png"))

def main():
    args = parse_args()
    configs = yaml_loader(args.configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join("output", configs["task"], "inference"), exist_ok=True)
    
    in_channels = int(configs["model"]['in_channels'])
    img_size = int(configs["model"]['img_size'])
    model = build_model(configs)
    summary(model, input_size=(1, in_channels, img_size, img_size))
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    torch.manual_seed(42)
    
    visualize_samples(model, configs, device, num_samples=16)
    