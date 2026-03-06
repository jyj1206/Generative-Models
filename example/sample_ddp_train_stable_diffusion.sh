conda activate gen_model
torchrun --nproc_per_node=4 train_stable_diffusion.py --config configs/StableDiffusion/ddpm_vqgan_celebA_HQ.yaml
