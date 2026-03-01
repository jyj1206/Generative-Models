import os
import time
import argparse
import yaml
import torch

from models.build import build_model, build_diffusion_scheduler
from utils.util_paths import append_timestamp, get_timestamp
from utils.util_visualization import save_diffusion_sampling_gif, generate_and_save_samples


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


def main():
    args = parse_args()
    configs = yaml_loader(args.configs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    output_dir = os.path.dirname(model_dir)
    configs["output_dir"] = output_dir
    inference_dir = os.path.join(output_dir, "inference")
    if configs.get("task") == "diffusion":
        diffuser = configs.get("diffusion", {}).get("diffuser")
        if diffuser:
            diffuser_name = diffuser.replace("_scheduler", "")
            inference_dir = os.path.join(inference_dir, diffuser_name)
    os.makedirs(inference_dir, exist_ok=True)
    model = build_model(configs)

    load_checkpoint(model, args.model_path, device)
    model.to(device)
    model.eval()

    task = configs["task"]
    if task == "vae":
        generator = model.decoder
    elif task == "gan":
        generator = model.netG
    else:
        generator = model

    if args.seed is not None:
        torch.manual_seed(args.seed)

    diffusion = None
    use_cfg = False
    if configs["task"] == "diffusion":
        diffusion, _ = build_diffusion_scheduler(configs, device)
        use_cfg = configs.get("diffusion", {}).get("use_cfg", False)  # Explicit CFG flag
        
        # Validate class_id if provided
        if args.class_id is not None:
            if not use_cfg:
                raise ValueError("class_id was provided but use_cfg=False in config.")
            num_classes = configs.get("dataset", {}).get("num_classes", None)
            if num_classes is None:
                raise ValueError("class_id requires dataset.num_classes to be set in config.")
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

    generate_and_save_samples(
        generator,
        configs,
        device,
        diffusion=diffusion if task == "diffusion" else None,
        sampling_steps=args.sampling_steps,
        use_cfg=use_cfg,
        class_id=args.class_id,
        guidance_scale=args.guidance_scale,
        num_samples=num_to_sample,
        save_to_file=True,
        save_dir=inference_dir,
        filename=save_filename,
        scale=args.scale,
    )

    elapsed = time.perf_counter() - start_time
    if grid_requested:
        print(f"Saved grid sample: {save_path}")
    else:
        print(f"Saved single sample: {save_path}")

    print(f"Inference time: {elapsed:.3f} sec")

    if configs["task"] == "diffusion" and args.diffusion_gif:
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


if __name__ == "__main__":
    main()