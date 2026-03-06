import os
import torch
import torch.distributed as dist


def set_visible_gpus(configs):
    gpu_ids = configs.get("run_time", {}).get("gpu_ids")
    
    if "CUDA_VISIBLE_DEVICES" not in os.environ and gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)


def setup_runtime():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if torch.cuda.is_available():
        if distributed:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            # single process
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return distributed, local_rank, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0