import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        """

        Args:
            time : Timesteps (Batch,)

        Returns:
            emb : Sinusoidal Position Embeddings (Batch, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input Image (Batch, C, H, W)
        Returns:
            out: feature map after Block (Batch, C, H, W)
        """
        return self.block(x)
        
        
class ResBlock(nn.Module):    
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out)
            )
        else:
            self.time_emb_proj = None
            
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups, dropout=dropout)
    
        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()
    
    def forward(self, x, time_emb=None):
        """
        
        Args:
            x: Input Image (Batch, C, H, W)
            time: Timesteps (Batch,)
            
        Returns:
            out: feature map after ResBlock (Batch, C, H, W)
        """
        h = self.block1(x)
        
        if self.time_emb_proj is not None and time_emb is not None:
            time_emb = self.time_emb_proj(time_emb)
            h = h + time_emb[:, :, None, None]
            
        h = self.block2(h)    
        
        x = self.res_conv(x)
        
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4): 
        super().__init__()
        self.num_heads = num_heads
        inner_dim = dim 
        self.scale = (dim // num_heads) ** -0.5
        
        self.norm = nn.GroupNorm(32, dim) 
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False) 
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        attn = q.transpose(-2, -1) @ k * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v.transpose(-2, -1)
        out = out.transpose(-2, -1).reshape(b, -1, h, w)
        out = self.to_out(out)
        return x + out
        
        
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        """
        Args:
            x: Input Image (Batch, C, H, W)

        Returns:
            x: Downsampled Image (Batch, C, H/2, W/2)
        """
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        """
        Args:
            x: Input Image (Batch, C, H, W)

        Returns:
            x: Upsampled Image (Batch, C, H*2, W*2)
        """
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, dim, init_dim=None, dim_mults=(1, 2, 2, 2), attn_layers=(16,), num_res_blocks = 2, dropout=0.0, in_channels=3, image_size=32):
        super().__init__()

        init_dim = dim if init_dim is None else init_dim 
        time_dim = dim * 4 

        self.init_conv = nn.Conv2d(in_channels, init_dim, 3, padding=1)
    
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )                
            
        # Down Block (Encoder)
        cur_channels = init_dim
        cur_ressolutions = image_size
                
        self.downs = nn.ModuleList([])
        skip_channels_list = [init_dim]
        
        for i in range(len(dim_mults)):
            dim_out = dim * dim_mults[i]
            
            use_attn = (cur_ressolutions in attn_layers)
            
            for _ in range(num_res_blocks):
                self.downs.append(nn.ModuleList([
                    ResBlock(cur_channels, dim_out, time_emb_dim=time_dim),
                    AttentionBlock(dim_out) if use_attn else nn.Identity()
                ]))
                cur_channels = dim_out
                skip_channels_list.append(cur_channels)
     
            if i != len(dim_mults) - 1:
                self.downs.append(Downsample(cur_channels))
                cur_ressolutions = cur_ressolutions // 2
                skip_channels_list.append(cur_channels)

        
        # Mid Block (Bottleneck)
        self.mid_block1 = ResBlock(cur_channels, cur_channels, time_emb_dim=time_dim)
        self.mid_attn = AttentionBlock(cur_channels)
        self.mid_block2 = ResBlock(cur_channels, cur_channels, time_emb_dim=time_dim)
        
        # Up Block (Decoder)
        self.ups = nn.ModuleList([])

        
        for i in reversed(range(len(dim_mults))):
            dim_out = dim * dim_mults[i]
            use_attn = (cur_ressolutions in attn_layers)

            for _ in range(num_res_blocks + 1):
                skip_channels = skip_channels_list.pop()
                self.ups.append(nn.ModuleList([
                    ResBlock(cur_channels + skip_channels, dim_out, time_emb_dim=time_dim),
                    AttentionBlock(dim_out) if use_attn else nn.Identity(),
                ]))
                cur_channels = dim_out
            
            if not i==0:
                self.ups.append(Upsample(cur_channels))
                cur_ressolutions = cur_ressolutions * 2
        
        # Final conv
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, dim), 
            nn.SiLU(),
            nn.Conv2d(dim, in_channels, 1)
        )
        
    def forward(self, x, time):
        """
        Args:
            x: Input Image (Batch, C, H, W)
            time: Timesteps (Batch,)
        Returns:
            out: Output noise prediction (Batch, C, H, W)
        """
        time_emb = self.time_mlp(time)
        
        x = self.init_conv(x)
        skip_connections = []
        
        skip_connections.append(x)
        
        # Downsample
        for down_block in self.downs:
            if isinstance(down_block, Downsample):
                x = down_block(x)
                skip_connections.append(x)
                continue
            res_block, attn_block = down_block
            x = res_block(x, time_emb)
            x = attn_block(x)
            skip_connections.append(x)
    
        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)
        
        # Upsample
        for up_block in self.ups:
            if isinstance(up_block, Upsample):
                x = up_block(x)
                continue
            res_block, attn_block = up_block
            skip_connection = skip_connections.pop()
            x = torch.cat((x, skip_connection), dim=1)
            x = res_block(x, time_emb)
            x = attn_block(x)

        x = self.final_conv(x)
        return x
        

if __name__ == "__main__":
    img_size = 256
    dim = 128
    in_channels = 3
    dim_mults = [1, 1, 2, 2, 4, 4] 
    attn_layers = [16]
    num_res_blocks = 2
    model = UNet(dim=dim, in_channels=in_channels, image_size=img_size, dim_mults=dim_mults, attn_layers=attn_layers, num_res_blocks=num_res_blocks)
  
    summary(model, input_size=[(4, 3, 256, 256), (4,)], depth=4)