import os
import math
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.build import build_dataset
from models.build import build_model, build_loss_function, build_optimizer
from utils.util_visualization import save_loss_curve, save_vae_recon_grid, save_vqvae_latent
from utils.util_save import save_vqvae_checkpoint
from utils.util_logger import setup_train_logger
from utils.util_paths import build_output_dir
from utils.util_ddp import set_visible_gpus, setup_runtime, cleanup_ddp, is_main


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VQ-VAE model (VQ regularization).")
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

    train_dataset, test_dataset = build_dataset(configs)

    sampler = None
    if configs.get('distributed', False):
        sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        sampler=sampler,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    return train_loader, test_loader


def compute_num_epochs(train_loader, configs):
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise ValueError("Train dataloader returned zero batches. Check dataset setup.")

    train_cfg = configs["train"]
    if "iterations" in train_cfg:
        iterations_target = int(train_cfg["iterations"])
        return max(1, math.ceil(iterations_target / steps_per_epoch))
    return int(train_cfg["epochs"])


def load_resume_state(models, optimizers, device, resume_path):
    vqvae_checkpoint = torch.load(resume_path, map_location=device)

    resume_name = os.path.basename(resume_path)
    if "vqvae_autoencoder" in resume_name:
        disc_resume_path = resume_path.replace("vqvae_autoencoder", "vqvae_discriminator")
    elif "vqvae_model" in resume_name:
        disc_resume_path = resume_path.replace("vqvae_model", "discriminator_model")
    else:
        raise ValueError("For VQ-VAE resume, pass an autoencoder checkpoint path (vqvae_autoencoder_*).")

    if not os.path.exists(disc_resume_path):
        raise FileNotFoundError(f"Matching discriminator checkpoint not found: {disc_resume_path}")

    disc_checkpoint = torch.load(disc_resume_path, map_location=device)

    models['vqvae'].load_state_dict(vqvae_checkpoint["vqvae_state_dict"])
    models['discriminator'].load_state_dict(disc_checkpoint["discriminator_state_dict"])
    optimizers['optimizer_g'].load_state_dict(vqvae_checkpoint["optimizer_g_state_dict"])
    optimizers['optimizer_d'].load_state_dict(disc_checkpoint["optimizer_d_state_dict"])

    start_epoch = int(vqvae_checkpoint.get("epoch", 0))
    iterations = int(vqvae_checkpoint.get("iterations", 0))
    return start_epoch, iterations


def calculate_adaptive_weight(nll_loss, g_loss, disc_weight, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * disc_weight
    return d_weight


def main():
    args = parse_args()
    configs = yaml_loader(args.config)
    set_visible_gpus(configs)
    distributed, local_rank, device = setup_runtime()
    configs['distributed'] = distributed

    output_dir = prepare_output_dir(configs, args.config, args.resume)
    logger = setup_train_logger(output_dir)
    if is_main():
        print(f"Logging to: {logger.log_path}")
    train_loader, test_loader = build_dataloaders(configs)
    has_test_loader = test_loader is not None
    if has_test_loader:
        os.makedirs(os.path.join(output_dir, "visualization", "test"), exist_ok=True)

    models = build_model(configs)
    criterions = build_loss_function(configs)
    optimizers = build_optimizer(models, configs)

    num_epochs = compute_num_epochs(train_loader, configs)

    train_loss_history = []
    disc_loss_history = []
    disc_loss_epochs = []
    test_loss_history = []
    start_epoch = 0
    iterations = 0

    model = models['vqvae'].to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    lpips_model = models['lpips'].eval().to(device)
    discriminator = models['discriminator'].to(device)

    recon_criterion = criterions['recon_criterion']
    discriminator_criterion = criterions['discriminator_criterion']

    optimizer_g = optimizers['optimizer_g']
    optimizer_d = optimizers['optimizer_d']

    if args.resume:
        start_epoch, iterations = load_resume_state(models, optimizers, device, args.resume)

    commitment_weight = float(configs["train"].get("commitment_weight", configs["model"].get("commitment_cost", 0.25)))
    codebook_weight = float(configs["train"].get("codebook_weight", 1.0))
    perceptual_weight = float(configs["train"].get("perceptual_weight", 1.0))
    disc_weight = float(configs["train"].get("disc_weight", 0.5))
    disc_step_start = int(configs["train"].get("disc_step_start", 15000))
    step_count = iterations

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        discriminator.train()

        model_train_loss_sum = 0.0
        discriminator_train_loss_sum = 0.0

        if distributed:
            train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", disable=not is_main())

        for inputs, _ in pbar:
            inputs = inputs.to(device)

            optimizer_g.zero_grad()

            step_count += 1

            model_output = model(inputs)
            outputs, reg_losses = model_output

            recon_loss = recon_criterion(outputs, inputs)
            lpips_loss = torch.mean(lpips_model(outputs, inputs))
            nll_loss = recon_loss + perceptual_weight * lpips_loss
            codebook_loss = reg_losses['codebook_loss']
            commitment_loss = reg_losses['commitment_loss']

            disc_factor = 1.0 if step_count > disc_step_start else 0.0

            g_loss = torch.zeros((), device=device)
            d_weight = torch.tensor(0.0, device=device)

            if disc_factor > 0.0:
                discriminator.eval()  # freeze BN running stats + disable dropout while computing G's adversarial loss
                discriminator.requires_grad_(False)   # don't accumulate grads for D params; only backprop to G through outputs
                disc_fake_pred = discriminator(outputs)
                g_loss = discriminator_criterion.generator_loss(disc_fake_pred)
                d_weight = calculate_adaptive_weight(nll_loss, g_loss, disc_weight, last_layer=model.decoder_conv_out.weight)
                discriminator.requires_grad_(True)
                discriminator.train()

            quantization_loss = codebook_loss + commitment_weight * commitment_loss
            loss = nll_loss + d_weight * disc_factor * g_loss + codebook_weight * quantization_loss
            loss.backward()
            optimizer_g.step()

            if disc_factor > 0.0:
                optimizer_d.zero_grad()
                fake = outputs
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(inputs)
                d_raw_loss = discriminator_criterion.discriminator_loss(disc_real_pred, disc_fake_pred)
                d_loss = disc_factor * d_raw_loss
                d_loss.backward()
                optimizer_d.step()
            else:
                d_loss = torch.zeros((), device=device)

            model_train_loss_sum += loss.item()
            discriminator_train_loss_sum += d_loss.item()
            iterations += 1
            if is_main():
                pbar.set_postfix({
                    "loss_g": f"{loss.item():.6f}",
                    "loss_d": f"{d_loss.item():.6f}",
                    "iter": iterations,
                })

        avg_train_loss = model_train_loss_sum / len(train_loader)
        avg_disc_train_loss = discriminator_train_loss_sum / len(train_loader)
        train_loss_history.append(avg_train_loss)
        if step_count > disc_step_start:
            disc_loss_history.append(avg_disc_train_loss)
            disc_loss_epochs.append(epoch)

        avg_test_loss = None
        if has_test_loader:
            model.eval()

            test_loss_sum = 0.0
            with torch.no_grad():
                for test_inputs, _ in test_loader:
                    test_inputs = test_inputs.to(device)
                    outputs, reg_losses = model(test_inputs)
                    test_loss = recon_criterion(outputs, test_inputs)
                    regularization_loss = codebook_weight * (reg_losses['codebook_loss'] + commitment_weight * reg_losses['commitment_loss'])
                    test_loss = test_loss + regularization_loss
                    test_loss_sum += test_loss.item()

            avg_test_loss = test_loss_sum / len(test_loader)
            test_loss_history.append(avg_test_loss)

        log_message = f"Epoch [{epoch}/{num_epochs}] Done. | G Loss: {avg_train_loss:.6f} | D Loss: {avg_disc_train_loss:.6f}"
        if avg_test_loss is not None:
            log_message += f" | Test Loss: {avg_test_loss:.6f}"
        if is_main():
            print(log_message)

            if epoch % 5 == 0:
                save_vae_recon_grid(model, configs, train_loader, device, epoch, train=True, scale=args.scale)
                save_vqvae_latent(model, configs, train_loader, device, epoch, train=True, scale=args.scale)
                if has_test_loader:
                    save_vae_recon_grid(model, configs, test_loader, device, epoch, train=False, scale=args.scale)
                    save_vqvae_latent(model, configs, test_loader, device, epoch, train=False, scale=args.scale)
                save_vqvae_checkpoint(models, optimizers, {"loss_g": avg_train_loss, "loss_d": avg_disc_train_loss}, configs, epoch, iterations)

    if is_main():
        model.eval()
        if has_test_loader:
            save_vae_recon_grid(model, configs, test_loader, device, train=False, scale=args.scale)
            save_vqvae_latent(model, configs, test_loader, device, train=False, scale=args.scale)
        save_loss_curve(configs, train_loss_history, disc_loss_history, "Generator Loss", "Discriminator Loss", x_history1=list(range(start_epoch + 1, start_epoch + len(train_loss_history) + 1)), x_history2=disc_loss_epochs)
        save_vqvae_checkpoint(models, optimizers, {"loss_g": avg_train_loss, "loss_d": avg_disc_train_loss}, configs, num_epochs, iterations, final=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
