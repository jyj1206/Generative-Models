conda activate gen_model
torchrun --nproc_per_node=4 train_diffusion.py --config configs/Diffusion/ddpm_cifar10.yaml
