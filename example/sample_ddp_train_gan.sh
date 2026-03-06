conda activate gen_model
torchrun --nproc_per_node=4 train_gan.py --config configs/GANs/vanila_gan_cifar10.yaml