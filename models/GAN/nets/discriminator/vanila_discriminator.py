import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, img_size=32):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        ds_size = img_size // 8

        self.fc = nn.Linear(256 * ds_size * ds_size, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.apply(self._init_weights)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x).squeeze(1)
        return x

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if 'Conv' in classname:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif 'BatchNorm' in classname:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
