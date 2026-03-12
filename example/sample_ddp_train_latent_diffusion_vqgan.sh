conda activate gen_model
torchrun --nproc_per_node=4 train_latent_diffusion.py --config configs/latent_diffusion/ddpm_vqgan_celebA_HQ.yaml
