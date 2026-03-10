conda activate gen_model
torchrun --nproc_per_node=4 train_vqgan_kl.py --config configs/VAE_GAN/vqgan_kl_celebA_HQ.yaml

