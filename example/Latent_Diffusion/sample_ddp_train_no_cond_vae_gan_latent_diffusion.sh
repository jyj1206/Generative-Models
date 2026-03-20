conda activate gen_model
torchrun --nproc_per_node=4 train_latent_diffusion.py --config configs/Latent_Diffusion/ddpm_vae_gan_no_cond_celebA_HQ.yaml
