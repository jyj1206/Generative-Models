import torch
import torch.nn as nn
from models.VAE.vae_loss import vae_kl_loss
from models.VAE.nets.blocks import DownBlock, MidBlock, UpBlock


class VAE(nn.Module):
    def __init__(self, num_heads=4, groups=32, num_down_layers=2, num_mid_layers=2, num_up_layers=2,
                 z_channels=4, in_channels=3):
        super().__init__()
        self.down_channels = [64, 128, 256, 256]
        self.mid_channels = [256, 256]
        self.down_sample = [True, True, True]
        self.num_down_layers = num_down_layers
        self.num_mid_layers = num_mid_layers
        self.num_up_layers = num_up_layers
        self.norm_channels = groups
        
        # self-attention
        self.num_heads = num_heads
        
        # latent dimmension
        self.z_channels = z_channels
        
        self.up_sample = list(reversed(self.down_sample))
        
        ############################### Encoder ##################################
        self.encoder_conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, stride=1, padding=1)
        
        self.encoder_downs = nn.ModuleList()
        for i in range(len(self.down_channels) - 1):
            self.encoder_downs.append(
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    self.down_sample[i],
                    self.num_down_layers,
                    self.norm_channels,
                )
            )
            
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(
                MidBlock(
                    self.mid_channels[i],
                    self.mid_channels[i + 1],
                    self.num_heads,
                    self.num_mid_layers,
                    self.norm_channels,
                )
            )
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2 * self.z_channels, kernel_size=3, stride=1, padding=1)
        
        # Latent Dimension is 2 * lantent dimension because we need to output both mean and logvar for reparameterization trick.
        self.pre_latent_conv = nn.Conv2d(2 * self.z_channels, 2 * self.z_channels, kernel_size=3, stride=1, padding=1)
        
        ################################## Decoder ##################################
        self.post_latent_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=3, stride=1, padding=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, stride=1, padding=1)
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(len(self.mid_channels) - 1)):
            self.decoder_mids.append(
                MidBlock(
                    self.mid_channels[i + 1],
                    self.mid_channels[i],
                    self.num_heads,
                    self.num_mid_layers,
                    self.norm_channels,
                )
            )
            
        self.decoder_ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.decoder_ups.append(
                UpBlock(
                    self.down_channels[i + 1],
                    self.down_channels[i],
                    self.up_sample[i],
                    self.num_up_layers,
                    self.norm_channels,
                )
            )
            
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], in_channels, kernel_size=3, stride=1, padding=1)
        
    def reparameterization_trick(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_downs:
            out = down(out)
            
        for mid in self.encoder_mids:
            out = mid(out)
        
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        
        out = self.encoder_conv_out(out)
        out = self.pre_latent_conv(out)

        kl_loss = vae_kl_loss(out)
    
        mean, logvar = torch.chunk(out, 2, dim=1)
        z = self.reparameterization_trick(mean, logvar)
        
        return z, kl_loss
    
    def decode(self, z):
        out = self.post_latent_conv(z)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_ups:
            out = up(out)
    
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out 
    
    
    def forward(self, x):
        z, kl_loss = self.encode(x)
        out = self.decode(z)
        return out, kl_loss

        
        

        