import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, img_size=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.feat_size = img_size // 8

        self.fc_mu = nn.Linear(128 * self.feat_size * self.feat_size, latent_dim)
        self.fc_logvar = nn.Linear(128 * self.feat_size * self.feat_size, latent_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
