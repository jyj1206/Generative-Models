conda activate gen_model
torchrun --nproc_per_node=4 train_vae.py --config configs/VAE/vanila_vae_cifar10.yaml
