
import torch
import torch.nn as nn

# reference
# 1. UNet paper https://arxiv.org/pdf/1505.04597.pdf
# 2. https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201

class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bnd1(self.relu(self.conv1(x)))
        x = self.bnd2(self.relu(self.conv2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = ConvBlock(n_in, n_out)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p
    
class Decoder(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv = ConvBlock(n_out*2, n_out)
        
    def forward(self, x, e):
        # up-conv 2x2
        x = self.deconv(x)
        # copy and crop
        x = torch.cat([x,e], axis=1)
        # conv 3x3, ReLU
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        # encoder
        self.enc1 = Encoder(3, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)
        
        # bottleneck
        self.conv = ConvBlock(512, 1024)
        
        # decoder
        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)
        
        # classifier
        self.outputs = nn.Conv2d(in_channels=64, out_channels=self.n_class, kernel_size=1, padding=0)
        
    def forward(self, x):
        
        # encoding (save the output before the pooling for later copy)
        x1,p1 = self.enc1(x)
        x2,p2 = self.enc2(p1)
        x3,p3 = self.enc3(p2)
        x4,p4 = self.enc4(p3)
        
        # bottleneck
        b = self.conv(p4)
        
        # decoding
        x = self.dec1(b, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        
        # classify
        outputs = self.outputs(x)
        
        return outputs   # size=(N, n_class, H, W)

        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
