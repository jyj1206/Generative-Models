import os
import math
import argparse
import yaml

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.build import build_dataset
from models.build import build_model, build_loss_function, build_optimizer
from utils.util_makegif import TrainRecorder
from utils.util_visualization import generate_and_save_samples, save_loss_curve
from utils.util_save import save_gan_checkpoint
from utils.util_logger import setup_train_logger
from utils.util_paths import build_output_dir
from utils.util_ddp import set_visible_gpus, setup_runtime, cleanup_ddp, is_main


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vanilla GAN model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
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

    sampler = None
    if configs.get("distributed", False):
        sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_resume_state(netG, netD, optimizer_g, optimizer_d, device, resume_path):
    checkpoint = torch.load(resume_path, map_location=device, weights_only=True)

    netG.load_state_dict(checkpoint["generator_state_dict"])
    netD.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    start_epoch = int(checkpoint.get("epoch", 0))
    iterations = int(checkpoint.get("iterations", 0))
    last_loss_g = float(checkpoint.get("loss_g", float("nan")))
    last_loss_d = float(checkpoint.get("loss_d", float("nan")))

    return start_epoch, iterations, last_loss_g, last_loss_d


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

    train_loader = build_dataloaders(configs)

    model = build_model(configs).to(device)
    netG = model.netG
    netD = model.netD

    if distributed:
        netG = DDP(netG, device_ids=[local_rank])
        netD = DDP(
            netD,
            device_ids=[local_rank],
            broadcast_buffers=False,
        )

    raw_netG = unwrap_model(netG)
    raw_netD = unwrap_model(netD)

    criterion = build_loss_function(configs)
    optimizer_g = build_optimizer(raw_netG, configs)
    optimizer_d = build_optimizer(raw_netD, configs)

    num_epochs = compute_num_epochs(train_loader, configs)
    target_iterations = int(configs["train"].get("iterations", 0))

    train_recorder = TrainRecorder(configs, device, num_samples=16, scale=args.scale)

    loss_g_history = []
    loss_d_history = []

    start_epoch = 0
    iterations = 0
    avg_loss_g = float("nan")
    avg_loss_d = float("nan")

    if args.resume:
        start_epoch, iterations, avg_loss_g, avg_loss_d = load_resume_state(
            raw_netG, raw_netD, optimizer_g, optimizer_d, device, args.resume
        )

    if target_iterations > 0 and iterations >= target_iterations:
        if is_main():
            print(
                f"Resume checkpoint already reached target iterations: "
                f"iterations={iterations}, target_iterations={target_iterations}"
            )
        cleanup_ddp()
        return

    if target_iterations == 0 and start_epoch >= num_epochs:
        if is_main():
            print(
                f"Resume checkpoint already reached target epochs: "
                f"start_epoch={start_epoch}, num_epochs={num_epochs}"
            )
        cleanup_ddp()
        return

    for epoch in range(start_epoch + 1, num_epochs + 1):
        netG.train()
        netD.train()

        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{num_epochs}]",
            disable=not is_main(),
        )

        epoch_loss_d_sum = 0.0
        epoch_loss_g_sum = 0.0
        epoch_sample_count = 0

        for real_images, _ in pbar:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            # --------------------
            # D step
            # --------------------
            for p in raw_netD.parameters():
                p.requires_grad = True
            netD.train()

            optimizer_d.zero_grad(set_to_none=True)

            z = torch.randn(batch_size, raw_netG.latent_dim, device=device)
            fake_images = netG(z)

            output_real = netD(real_images)
            output_fake = netD(fake_images.detach())

            label_real = torch.full_like(output_real, 0.9)
            label_fake = torch.zeros_like(output_fake)

            loss_d_real = criterion(output_real, label_real)
            loss_d_fake = criterion(output_fake, label_fake)
            loss_d = 0.5 * (loss_d_real + loss_d_fake)

            loss_d.backward()
            optimizer_d.step()

            # --------------------
            # G step
            # --------------------
            for p in raw_netD.parameters():
                p.requires_grad = False
            raw_netD.eval()

            optimizer_g.zero_grad(set_to_none=True)

            output_gen = raw_netD(fake_images)
            label_gen = torch.full_like(output_gen, 0.9)
            loss_g = criterion(output_gen, label_gen)

            loss_g.backward()
            optimizer_g.step()

            # --------------------
            # logging
            # --------------------
            epoch_loss_d_sum += loss_d.item() * batch_size
            epoch_loss_g_sum += loss_g.item() * batch_size
            epoch_sample_count += batch_size
            iterations += 1

            if is_main():
                pbar.set_postfix({
                    "loss_D": f"{loss_d.item():.6f}",
                    "loss_G": f"{loss_g.item():.6f}",
                    "iter": iterations,
                })

            if target_iterations > 0 and iterations >= target_iterations:
                break

        loss_d_sum_tensor = torch.tensor(epoch_loss_d_sum, device=device)
        loss_g_sum_tensor = torch.tensor(epoch_loss_g_sum, device=device)
        sample_count_tensor = torch.tensor(epoch_sample_count, device=device, dtype=torch.long)

        if distributed:
            dist.all_reduce(loss_d_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_g_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count_tensor, op=dist.ReduceOp.SUM)

        total_samples = max(1, int(sample_count_tensor.item()))
        avg_loss_d = loss_d_sum_tensor.item() / total_samples
        avg_loss_g = loss_g_sum_tensor.item() / total_samples

        loss_d_history.append(avg_loss_d)
        loss_g_history.append(avg_loss_g)

        if is_main():
            print(
                f"Epoch [{epoch}/{num_epochs}] Done. | "
                f"Loss D: {avg_loss_d:.6f} | Loss G: {avg_loss_g:.6f}"
            )

            netG.eval()
            train_recorder.record_frame(raw_netG, epoch)
            netG.train()

            if epoch % 25 == 0:
                generate_and_save_samples(
                    raw_netG,
                    configs,
                    device,
                    num_samples=16,
                    epoch=epoch,
                    scale=args.scale,
                )
                save_gan_checkpoint(
                    model,
                    optimizer_g,
                    optimizer_d,
                    avg_loss_g,
                    avg_loss_d,
                    configs,
                    epoch,
                    iterations,
                )

        if target_iterations > 0 and iterations >= target_iterations:
            break

    if is_main():
        netG.eval()
        netD.eval()

        generate_and_save_samples(
            raw_netG,
            configs,
            device,
            num_samples=16,
            scale=args.scale,
        )
        train_recorder.save_gif(filename="training_process.gif", duration=100)
        save_loss_curve(
            configs,
            loss_g_history,
            loss_d_history,
            "Generator Loss",
            "Discriminator Loss",
        )
        save_gan_checkpoint(
            model,
            optimizer_g,
            optimizer_d,
            avg_loss_g,
            avg_loss_d,
            configs,
            epoch if "epoch" in locals() else start_epoch,
            iterations,
            final=True,
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()