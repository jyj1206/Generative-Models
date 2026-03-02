import torch.nn as nn
from torchinfo import summary
from models.GAN.nets.generator.vanila_generator import Generator
from models.GAN.nets.discriminator.vanila_discriminator import Discriminator


class VanillaGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(VanillaGAN, self).__init__()
        self.netG = generator
        self.netD = discriminator

    def forward(self, z):
        generated_data = self.netG(z)
        return generated_data
            

if __name__ == "__main__":  
    latent_dim = 100
    img_size = 32
    in_channels = 3

    generator = Generator(out_channels=in_channels, latent_dim=latent_dim, img_size=img_size)
    discriminator = Discriminator(in_channels=in_channels, img_size=img_size)
    model = VanillaGAN(generator, discriminator)

    summary(model.netG, input_size=(4, latent_dim), depth=4)
    summary(model.netD, input_size=(4, in_channels, img_size, img_size), depth=4)