import torch
import torch.nn as nn
from torchinfo import summary
from models.VAE.nets.encoder.vanila_encoder import Encoder
from models.VAE.nets.decoder.vanila_decoder import Decoder


class VanillaVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super(VanillaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        
    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization_trick(mu, logvar)
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar
    
    
if __name__ == "__main__":
    IMG_SIZE = 32
    LATENT_DIM = 100
    
    encoder = Encoder(in_channels=3, latent_dim=LATENT_DIM, img_size=IMG_SIZE)
    decoder = Decoder(out_channels=3, latent_dim=LATENT_DIM, img_size=IMG_SIZE)
    
    model = VanillaVAE(encoder, decoder, latent_dim=LATENT_DIM)
    
    summary(model, input_size=(4, 3, IMG_SIZE, IMG_SIZE), depth=4)