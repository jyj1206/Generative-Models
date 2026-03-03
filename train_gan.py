import os
import math
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.build import build_dataset
from models.build import build_model, build_loss_function, build_optimizer
from utils.util_makegif import TrainRecorder
from utils.util_visualization import generate_and_save_samples, save_loss_curve
from utils.util_save import save_gan_checkpoint
from utils.util_logger import setup_train_logger
from utils.util_paths import build_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vanilla GAN model.")
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


def build_dataloaders(configs):
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


def load_resume_state(model, optimizer_g, optimizer_d, device, resume_path):
    checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
    model.netG.load_state_dict(checkpoint["generator_state_dict"])
    model.netD.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0))
    iterations = int(checkpoint.get("iterations", 0))
    return start_epoch, iterations


def main():
    args = parse_args()
    configs = yaml_loader(args.config)

    output_dir = prepare_output_dir(configs, args.config, args.resume)
    logger = setup_train_logger(output_dir)
    print(f"Logging to: {logger.log_path}")
    train_loader = build_dataloaders(configs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(configs).to(device)

    criterion = build_loss_function(configs)
    optimizer_g = build_optimizer(model.netG, configs)
    optimizer_d = build_optimizer(model.netD, configs)
    num_epochs = compute_num_epochs(train_loader, configs)
    train_recorder = TrainRecorder(configs, device, num_samples=16, scale=args.scale)

    loss_g_history = []
    loss_d_history = []
    start_epoch = 0
    iterations = 0

    if args.resume:
        start_epoch, iterations = load_resume_state(model, optimizer_g, optimizer_d, device, args.resume)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        loss_d_sum = 0.0
        loss_g_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]")

        for real_images, _ in pbar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            optimizer_d.zero_grad()
            z = torch.randn(batch_size, model.netG.latent_dim, device=device)
            fake_images = model.netG(z)
            label_real = torch.ones(batch_size, device=device) * 0.9
            label_fake = torch.zeros(batch_size, device=device)

            output_real = model.netD(real_images)
            output_fake = model.netD(fake_images.detach())

            loss_d_real = criterion(output_real, label_real)
            loss_d_fake = criterion(output_fake, label_fake)
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            output = model.netD(fake_images)
            loss_g = criterion(output, label_real)
            loss_g.backward()
            optimizer_g.step()

            loss_d_sum += loss_d.item()
            loss_g_sum += loss_g.item()
            iterations += 1
            pbar.set_postfix({
                "loss_D": f"{loss_d.item():.4f}",
                "loss_G": f"{loss_g.item():.4f}",
                "iter": iterations,
            })

        avg_loss_d = loss_d_sum / len(train_loader)
        avg_loss_g = loss_g_sum / len(train_loader)
        loss_d_history.append(avg_loss_d)
        loss_g_history.append(avg_loss_g)

        print(
            f"Epoch [{epoch}/{num_epochs}] Done. | Loss D: {avg_loss_d:.4f} | "
            f"Loss G: {avg_loss_g:.4f}"
        )

        model.netG.eval()
        train_recorder.record_frame(model.netG, epoch)
        model.netG.train()

        if epoch % 25 == 0:
            generate_and_save_samples(model.netG, configs, device, num_samples=16, epoch=epoch, scale=args.scale)
            save_gan_checkpoint(model, optimizer_g, optimizer_d, avg_loss_g, avg_loss_d, configs, epoch, iterations)

    model.eval()
    generate_and_save_samples(model.netG, configs, device, num_samples=16, scale=args.scale)
    train_recorder.save_gif(filename="training_process.gif", duration=100)
    save_loss_curve(configs, loss_g_history, loss_d_history, "Generator Loss", "Discriminator Loss")
    save_gan_checkpoint(model, optimizer_g, optimizer_d, avg_loss_g, avg_loss_d, configs, num_epochs, iterations, final=True)


if __name__ == "__main__":
    main()
