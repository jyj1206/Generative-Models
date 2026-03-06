conda activate gen_model
torchrun --nproc_per_node=4 train_vqvae_kl.py --config configs/VAE/vqvae_kl_celebA_HQ.yaml

