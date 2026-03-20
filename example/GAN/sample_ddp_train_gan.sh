conda activate gen_model
torchrun --nproc_per_node=4 train_gan.py --config configs/GAN/vanila_gan_cifar10.yaml