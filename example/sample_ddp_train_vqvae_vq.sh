conda activate gen_mode
torchrun --nproc_per_node=4 --master_port=12350 train_vqvae_vq.py --config configs/VAE/vqvae_vq_celebA_HQ.yaml
