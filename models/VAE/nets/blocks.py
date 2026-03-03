import torch.nn as nn


class DownBlock(nn.Module):
    r"""
    Down conv block.
    Sequence:
    1. ResNet-like block
    2. Downsample
    """

    def __init__(self, in_channels, out_channels, down_sample, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, 4, 2, 1) if self.down_sample else nn.Identity()

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with self-attention.
    Sequence:
    1. ResNet-like block
    2. Self-attention + ResNet-like block (repeated)
    """

    def __init__(self, in_channels, out_channels, num_heads, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers + 1)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
        )
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )

    def forward(self, x):
        out = self.resnet_conv_first[0](x)
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](x)

        for i in range(self.num_layers):
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UpBlock(nn.Module):
    r"""
    Up conv block.
    Sequence:
    1. Upsample
    2. ResNet-like block
    """

    def __init__(self, in_channels, out_channels, up_sample, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        
        # TODO: test용
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1) if self.up_sample else nn.Identity()
        
        # if self.up_sample:
        #     self.up_sample_conv = nn.Sequential(
        #         nn.Upsample(scale_factor=2, mode="nearest"),
        #         nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        #     )
        # else:
        #     self.up_sample_conv = nn.Identity()

    def forward(self, x):
        out = self.up_sample_conv(x)

        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        return out