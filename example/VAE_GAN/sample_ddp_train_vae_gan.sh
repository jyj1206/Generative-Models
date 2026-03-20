conda activate gen_model
torchrun --nproc_per_node=4 train_vae_gan.py --config configs/VAE_GAN/vae_gan_celebA_HQ.yaml

