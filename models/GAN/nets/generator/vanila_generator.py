import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, out_channels=3, latent_dim=100, img_size=32):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.feat_size = img_size // 8

        self.fc = nn.Linear(latent_dim, 256 * self.feat_size * self.feat_size)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.feat_size, self.feat_size)
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.tanh(self.deconv3(x))
        return x

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
