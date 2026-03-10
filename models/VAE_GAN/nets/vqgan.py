import torch
import torch.nn as nn
import torch.nn.functional as F
from models.VAE.nets.blocks import DownBlock, MidBlock, UpBlock


class VQGAN(nn.Module):
    def __init__(self, num_heads=4, groups=32, num_down_layers=2, num_mid_layers=2, num_up_layers=2,
                 z_channels=4, codebook_size=8192, in_channels=3):
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
        self.codebook_size = codebook_size
        
        self.up_sample = list(reversed(self.down_sample))
        
        
        ################################## Encoder ##################################
        self.encoder_conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, stride=1, padding=1)
        
        # Downblock + Midblock
        self.encoder_donws = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_donws.append(
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
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, stride=1, padding=1)
        
        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        
        # Codebook
        self.codebook = nn.Embedding(self.codebook_size, self.z_channels)
        
        ################################## Decoder ##################################
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
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
        
    
    def quantize(self, x):
        B, C, H, W = x.shape
        
        # B, C, H, W -> B, H, W, C
        x_perm = x.permute(0, 2, 3, 1)
        
        # B, H, W, C -> B, H*W, C
        x_flat = x_perm.reshape(B, H * W, C)

        # codebook: [K, C]
        e = self.codebook.weight
        k = e.size(0)    
        
        # ---- squared L2 distance: ||x-e||^2 = ||x||^2 + ||e||^2 - 2 x·e
        # x2: [B, HW, 1]
        x2 = torch.sum(x_flat ** 2, dim=-1, keepdim=True)
        # e2 : [1, 1, k]
        e2 = torch.sum(e ** 2, dim=-1).view(1, 1, k)
        # xe: [B, HW, K] (x: (1, c), e: (1, c) -> x·e.T: (1, 1))
        xe = torch.matmul(x_flat, e.t())
        
        # [B, HW, K]
        dist2 = x2 + e2 - 2 * xe
        
        # [B, HW]
        min_encoding_indices = torch.argmin(dist2, dim=-1)
        
        quant_flat = F.embedding(min_encoding_indices, e)
        
        commitment_loss = F.mse_loss(quant_flat.detach(), x_flat)
        codebook_loss = F.mse_loss(quant_flat, x_flat.detach())
        
        quantize_losses = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
        }
        
        quant_flat_st = x_flat + (quant_flat - x_flat).detach()  # dzq / dze = 1
        
        quant_out = quant_flat_st.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        return quant_out, quantize_losses
    
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_donws:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        z, quant_losses = self.quantize(out)
        return z, quant_losses
    
    
    def decode(self, z):
        out = self.post_quant_conv(z)
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
        z, quant_losses = self.encode(x)
        out = self.decode(z)
        return out, quant_losses


class VQGANInterface(VQGAN):
    def __init__(self, embed_dim, *args, **kwargs):
        kwargs.setdefault("z_channels", embed_dim)
        super().__init__(*args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_donws:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        return out

    def decode(self, h, force_not_quantize=False):
        if not force_not_quantize:
            quant, _ = self.quantize(h)
        else:
            quant = h
        dec = super().decode(quant)
        return dec