import inspect
import os
import torch
import torch.nn as nn
import torchvision
from collections import namedtuple


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class vgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        
        # Load pretrained vgg model
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.num_slices = 5
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
                
    def forward(self, x):
        x = self.slice1(x)
        x_relu1_2 = x
        x = self.slice2(x)
        x_relu2_2 = x
        x = self.slice3(x)
        x_relu3_3 = x
        x = self.slice4(x)
        x_relu4_3 = x
        x = self.slice5(x)
        x_relu5_3 = x
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(x_relu1_2, x_relu2_2, x_relu3_3, x_relu4_3, x_relu5_3)
        return out
                
                
class LPIPS(nn.Module):
    def __init__(self, net='vgg', version='0.1', use_dropout=True, device='cpu'):
        super().__init__()
        self.version = version
        self.scaling_layer = ScalingLayer()
        
        self.channels = [64, 128, 256, 512, 512]
        self.len_channels = len(self.channels)
        self.net = vgg16(pretrained=True, requires_grad=False)
        
        self.lin0 = NetLinearLayer(self.channels[0], use_dropout=use_dropout)
        self.lin1 = NetLinearLayer(self.channels[1], use_dropout=use_dropout)
        self.lin2 = NetLinearLayer(self.channels[2], use_dropout=use_dropout)
        self.lin3 = NetLinearLayer(self.channels[3], use_dropout=use_dropout)
        self.lin4 = NetLinearLayer(self.channels[4], use_dropout=use_dropout)


        model_path = os.path.abspath(
            os.path.join(inspect.getfile(self.__init__), '..', f'weights/v{version}/{net}.pth')
        )
        print(f'Loading model from: {model_path}')

        self.load_state_dict(torch.load(model_path, map_location=device), strict = False)
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
            
    def forward(self, input1, input2, normalize=False):
        # Sclae the inputs to -1 to +1 range if needed
        if normalize:
            input1 = self.scaling_layer(input1)
            input2 = self.scaling_layer(input2)
            
        input1_vgg, input2_vgg = self.scaling_layer(input1), self.scaling_layer(input2)
        
        out1, out2 = self.net(input1_vgg), self.net(input2_vgg)
        features1, features2, diffs = {}, {}, {}
        
        for kk in range(self.len_channels):
            features1[kk], features2[kk] = torch.nn.functional.normalize(out1[kk], dim=1), torch.nn.functional.normalize(out2[kk], dim=1)
            diffs[kk] = (features1[kk] - features2[kk]) ** 2
            
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.len_channels)]
        
        val = sum(res)
        
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Imagenet normalization for (0-1)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        
        self.register_buffer('shift', torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])
    
    def forward(self, x):
        return (x - self.shift) / self.scale
    
    
class NetLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
    def forward(self, x):
        out = self.model(x)
        return out