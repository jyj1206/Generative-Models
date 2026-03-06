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
from utils.util_makegif import TrainRecorder
from utils.util_visualization import generate_and_save_samples, save_loss_curve, save_vae_recon_grid
from utils.util_save import save_vae_checkpoint
from utils.util_logger import setup_train_logger
from utils.util_paths import build_output_dir
from utils.util_ddp import set_visible_gpus, setup_runtime, cleanup_ddp, is_main


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Vanilla VAE model.")
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


def load_resume_state(model, optimizer, device, resume_path):
    checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0))
    iterations = int(checkpoint.get("iterations", 0))
    return start_epoch, iterations


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

    model = build_model(configs).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    criterion = build_loss_function(configs)
    optimizer = build_optimizer(model, configs)
    num_epochs = compute_num_epochs(train_loader, configs)
    train_recorder = TrainRecorder(configs, device, num_samples=16, scale=args.scale)

    train_loss_history = []
    test_loss_history = []
    start_epoch = 0
    iterations = 0

    if args.resume:
        start_epoch, iterations = load_resume_state(model, optimizer, device, args.resume)

    for epoch in range(start_epoch + 1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]", disable=not is_main())

        for inputs, _ in pbar:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs, epoch=epoch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            iterations += 1
            if is_main():
                pbar.set_postfix({"loss": f"{loss.item():.6f}", "iter": iterations})

        avg_train_loss = train_loss_sum / len(train_loader)
        train_loss_history.append(avg_train_loss)

        avg_test_loss = None
        if has_test_loader:
            model.eval()

            test_loss_sum = 0.0
            with torch.no_grad():
                for test_inputs, _ in test_loader:
                    test_inputs = test_inputs.to(device)
                    outputs = model(test_inputs)
                    test_loss = criterion(outputs, test_inputs, epoch=epoch)
                    test_loss_sum += test_loss.item()

            avg_test_loss = test_loss_sum / len(test_loader)
            test_loss_history.append(avg_test_loss)

        log_message = f"Epoch [{epoch}/{num_epochs}] Done. | Train Loss: {avg_train_loss:.6f}"
        if avg_test_loss is not None:
            log_message += f" | Test Loss: {avg_test_loss:.6f}"
        if is_main():
            print(log_message)

            model.decoder.eval()
            train_recorder.record_frame(model.decoder, epoch)
            model.train()

            if epoch % 20 == 0:
                generate_and_save_samples(model.decoder, configs, device, num_samples=16, epoch=epoch, scale=args.scale)
                save_vae_recon_grid(model, configs, train_loader, device, epoch, train=True, scale=args.scale)
                if has_test_loader:
                    save_vae_recon_grid(model, configs, test_loader, device, epoch, train=False, scale=args.scale)
                save_vae_checkpoint(model, optimizer, avg_train_loss, configs, epoch, iterations)

    if is_main():
        model.eval()
        generate_and_save_samples(model.decoder, configs, device, num_samples=16, scale=args.scale)
        train_recorder.save_gif(filename="training_process.gif", duration=100)
        if has_test_loader:
            save_vae_recon_grid(model, configs, test_loader, device, scale=args.scale)
            save_loss_curve(configs, train_loss_history, test_loss_history)
        else:
            save_loss_curve(configs, train_loss_history)
        save_vae_checkpoint(model, optimizer, avg_train_loss, configs, num_epochs, iterations, final=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
