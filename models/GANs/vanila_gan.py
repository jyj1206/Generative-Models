import torch.nn as nn


class VanillaGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(VanillaGAN, self).__init__()
        self.netG = generator
        self.netD = discriminator

    def forward(self, z):
        generated_data = self.netG(z)
        return generated_data
    

class Generator(nn.Module):
    def __init__(self, out_channels=3, latent_dim=100, img_size=32):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim  
        self.feat_size = img_size // 8 
        
        self.fc = nn.Linear(latent_dim, 256 * self.feat_size * self.feat_size)
        
        # 4x4 -> 8x8
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128) 
        
        # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64) 
        
        # 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh() 
          
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.feat_size, self.feat_size)
        
        x = self.relu(self.bn1(self.deconv1(x))) 
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.tanh(self.deconv3(x))           
        
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        # 32x32 -> 16x16
        self.conv1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        
        # 16x16 -> 8x8
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128) 
        
        # 8x8 -> 4x4
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256) 
        # 4x4 -> 1x1 (진짜/가짜 스칼라값)
        self.final_conv = nn.Conv2d(256, 1, 4, stride=1, padding=0)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x))) # Conv -> BN -> LeakyReLU
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.final_conv(x)
        
        x = x.view(-1, 1).squeeze(1)
        
        return x 