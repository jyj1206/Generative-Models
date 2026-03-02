import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=128, img_size=32):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim

        self.feat_size = img_size // 8

        self.fc = nn.Linear(latent_dim, 128 * self.feat_size * self.feat_size)

        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1)

        self.relu = nn.ReLU()

        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)

        x = x.view(x.size(0), 128, self.feat_size, self.feat_size)

        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))

        x = self.activation(self.deconv3(x))

        return x
