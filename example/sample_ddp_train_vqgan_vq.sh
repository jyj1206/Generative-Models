conda activate gen_model
torchrun --nproc_per_node=4 --master_port=12350 train_vqgan_vq.py --config configs/VAE_GAN/vqgan_vq_celebA_HQ.yaml
