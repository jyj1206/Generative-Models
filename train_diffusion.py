import os
import shutil
import math
import copy
import argparse
import yaml

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from datasets.build import build_dataset
from models.build import build_model, build_optimizer, build_diffusion_scheduler
from utils.util_ema import ema_update
from utils.util_visualization import (
    generate_and_save_samples,
    save_diffusion_sampling_gif,
    save_loss_curve,
)
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
        sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, sampler


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
    ema_model.eval()
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


def parse_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        if len(batch) == 1:
            return batch[0], None
    return batch, None


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

    train_loader, train_sampler = build_dataloader(configs)

    base_model = build_model(configs).to(device)
    ema_model, ema_decay = setup_ema_model(base_model, configs, device)

    model = base_model
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = build_optimizer(model, configs)
    diffusion, num_timesteps = build_diffusion_scheduler(configs, device)
    num_epochs = compute_num_epochs(train_loader, configs)

    diffusion_cfg = configs.get("diffusion", {})
    use_cfg = diffusion_cfg.get("use_cfg", False)
    p_uncond = float(diffusion_cfg.get("p_uncond", 0.0)) if use_cfg else 0.0

    if use_cfg:
        if configs.get("dataset", {}).get("num_classes") is None:
            raise ValueError("Classifier-free guidance requires dataset.num_classes in the config.")
        if getattr(diffusion, "null_token_idx", None) is None:
            raise ValueError("Classifier-free guidance requires diffusion.null_token_idx to be set.")

    ema_enabled = ema_model is not None

    train_loss_history = []
    start_epoch = 0
    iterations = 0
    last_epoch = 0
    last_avg_train_loss = float("nan")

    if args.resume:
        start_epoch, iterations = load_resume_state(
            model,
            optimizer,
            device,
            args.resume,
            ema_model=ema_model,
            resume_ema_path=args.resume_ema,
        )
        last_epoch = start_epoch

    for epoch in range(start_epoch + 1, num_epochs + 1):
        last_epoch = epoch
        model.train()

        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{num_epochs}]",
            disable=not is_main(),
        )

        for batch in pbar:
            inputs, labels = parse_batch(batch)
            inputs = inputs.to(device, non_blocking=True)

            model_kwargs = None
            if use_cfg:
                if labels is None:
                    raise ValueError("CFG is enabled, but the dataset did not return labels.")
                labels = labels.to(device, non_blocking=True)

                if p_uncond > 0.0:
                    drop_mask = torch.rand(labels.shape, device=device) < p_uncond
                    labels = labels.clone()
                    labels[drop_mask] = diffusion.null_token_idx

                model_kwargs = {"y": labels}

            optimizer.zero_grad(set_to_none=True)

            t = torch.randint(0, num_timesteps, (inputs.size(0),), device=device).long()
            loss = diffusion.p_losses(model, inputs, t, model_kwargs=model_kwargs)

            loss.backward()
            optimizer.step()

            if ema_enabled:
                ema_update(unwrap_model(model), ema_model, ema_decay)

            batch_size = inputs.size(0)
            epoch_loss_sum += loss.detach().item() * batch_size
            epoch_sample_count += batch_size
            iterations += 1

            if is_main():
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "iter": iterations,
                })

        if epoch_sample_count == 0:
            break

        loss_sum_tensor = torch.tensor(epoch_loss_sum, device=device, dtype=torch.float32)
        count_tensor = torch.tensor(epoch_sample_count, device=device, dtype=torch.float32)

        if distributed:
            dist.all_reduce(loss_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        avg_train_loss = (loss_sum_tensor / count_tensor.clamp_min(1.0)).item()
        last_avg_train_loss = avg_train_loss
        train_loss_history.append(avg_train_loss)

        if is_main():
            print(f"Epoch [{epoch}/{num_epochs}] Done. | Train Loss: {avg_train_loss:.6f}")

            reference_model_to_use = ema_model if ema_enabled else unwrap_model(model)

            if epoch % 25 == 0:
                reference_model_to_use.eval()
                generate_and_save_samples(
                    reference_model_to_use,
                    configs,
                    device,
                    diffusion=diffusion,
                    num_samples=16,
                    epoch=epoch,
                    scale=args.scale,
                    use_cfg=use_cfg,
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
        reference_model_to_use = ema_model if ema_enabled else unwrap_model(model)
        reference_model_to_use.eval()

        generate_and_save_samples(
            reference_model_to_use,
            configs,
            device,
            diffusion=diffusion,
            num_samples=16,
            scale=args.scale,
            use_cfg=use_cfg,
        )
        save_diffusion_sampling_gif(
            reference_model_to_use,
            diffusion,
            configs,
            num_samples=16,
            capture_interval=20,
            scale=args.scale,
            use_cfg=use_cfg,
        )
        save_loss_curve(configs, train_loss_history)
        save_diffusion_checkpoint(
            unwrap_model(model),
            optimizer,
            last_avg_train_loss,
            configs,
            last_epoch,
            iterations,
            final=True,
            ema_model=ema_model if ema_enabled else None,
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()