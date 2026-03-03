import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

class PatchGANDiscriminator(nn.Module):
    """
    Pix2pix-style 70x70 PatchGAN discriminator (default for 256x256 -> [B,1,30,30]).
    - strides: 2/2/2/1/1
    - kernels: 4
    - paddings: 1
    - channels: 64/128/256/512/1
    """
    def __init__(self,
                 in_channels=3,
                 conv_channels=(64, 128, 256, 512),
                 kernels=(4, 4, 4, 4, 4),
                 strides=(2, 2, 2, 1, 1),
                 paddings=(1, 1, 1, 1, 1),
                 init_weights=False):
        super().__init__()

        layers_dim = [in_channels] + list(conv_channels) + [1]
        n_convs = len(layers_dim) - 1

        assert len(kernels) == n_convs, f"kernels 길이({len(kernels)}) != conv 개수({n_convs})"
        assert len(strides) == n_convs, f"strides 길이({len(strides)}) != conv 개수({n_convs})"
        assert len(paddings) == n_convs, f"paddings 길이({len(paddings)}) != conv 개수({n_convs})"

        self.layers = nn.ModuleList()
        last_idx = n_convs - 1

        for i in range(n_convs):
            in_ch, out_ch = layers_dim[i], layers_dim[i + 1]
            is_first = (i == 0)
            is_last  = (i == last_idx)

            conv_bias = is_first or is_last

            block = [
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=conv_bias),
            ]

            # pix2pix 관례: 첫/마지막 conv에는 norm 없음
            if (not is_first) and (not is_last):
                block.append(nn.BatchNorm2d(out_ch))

            # 마지막은 logits 맵 출력이므로 activation 없음
            if not is_last:
                block.append(nn.LeakyReLU(0.2, inplace=True))

            self.layers.append(nn.Sequential(*block))

        if init_weights:
            self.apply(weights_init)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


# quick check
if __name__ == "__main__":
    D = PatchGANDiscriminator()
    x = torch.randn(2, 3, 256, 256)
    y = D(x)
    print(y.shape)  # torch.Size([2, 1, 30, 30])