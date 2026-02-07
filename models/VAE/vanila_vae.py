import torch
import torch.nn as nn
from torchinfo import summary


class VanillaVAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super(VanillaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        
        
    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization_trick(mu, logvar)
        x_hat = self.decoder(z)
        
        return x_hat, mu, logvar
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, img_size = 32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        
        feat_size = img_size // 8
    
        self.fc_mu = nn.Linear(128 * feat_size * feat_size, latent_dim)
        self.fc_logvar = nn.Linear(128 * feat_size * feat_size, latent_dim)
        
        self.relu = nn.ReLU()
         
    
    def forward(self, x):
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
        

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=128, img_size = 32, activation='sigmoid'):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim  # 추가
        feat_size = img_size // 8
        
        self.fc = nn.Linear(latent_dim, 128 * feat_size * feat_size)
        
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1)
        
        self.relu = nn.ReLU()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()
        
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        
        x = self.activation(self.deconv3(x))
        
        return x
    
    
if __name__ == "__main__":

    encoder = Encoder(in_channels=3, latent_dim=128, img_size=32)
    decoder = Decoder(out_channels=3, latent_dim=128, img_size=32, activation='sigmoid')
    model = VanillaVAE(encoder, decoder, latent_dim=128)
    
    summary(model, input_size=(4, 3, 32, 32), depth=4)