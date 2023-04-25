import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# reference
# 1. UNet paper https://arxiv.org/pdf/1505.04597.pdf
# 2. https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
# 3. Torchvision ResNet code https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

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

class ResEncoder(nn.Module):
    def __init__(self): #, enc_size):
        super().__init__()
            
        self.pretrained_resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.pretrained_resnet50.parameters():
            param.requires_grad = False
            
        #print(list(self.pretrained_resnet50.children()))
        
        # ResNet50 Architecture
        # -- conv1, bn1, relu, maxpool (input)
        # -- layer1~4 (Sequential x 4) 
        # layer1-0: (conv, bn)x3, relu, downsample(conv, bn)
        # layer1-1~2: (conv, bn)x3, relu
        # layer2-0: (conv, bn)x3, relu, downsample(conv, bn)
        # layer2-1~3: (conv, bn)x3, relu
        # layer3-0: (conv, bn)x3, relu, downsample(conv, bn)
        # layer3-1~5: (conv, bn)x3, relu
        # layer4-0: (conv, bn)x3, relu, downsample(conv, bn)
        # layer4-1~2: (conv, bn)x3, relu
        # -- avgpool: AdaptiveAvgPool
        # -- fc: linear (2048, 1000)
        
        self.in_conv = self.pretrained_resnet50.conv1
        self.in_bn = self.pretrained_resnet50.bn1
        self.in_relu = self.pretrained_resnet50.relu
        self.in_maxpool = self.pretrained_resnet50.maxpool
        
        self.layers = nn.ModuleList([self.pretrained_resnet50.layer1, self.pretrained_resnet50.layer2, self.pretrained_resnet50.layer3, self.pretrained_resnet50.layer4])        
        
        #num_resout = self.pretrained_resnet50.fc.in_features
        #self.pretrained_resnet50.fc = nn.Linear(num_resout, enc_size)
                    
    def forward(self, img):

        # img: [batch_size x 3 x 256 x 256]
        x_s = []

        x = self.in_conv(img)
        x = self.in_bn(x)
        # output to use for unet
        x = self.in_relu(x)
        x_s.append(x)
        # x_s[0]: [batch_size x 64 x 128 x 128]

        x = self.in_maxpool(x)
        # => x: [batch_size x 64 x 64 x 64]
        
        for layer in self.layers:
            x = layer(x)
            x_s.append(x)            
        # x_s[1]: [batch_size x 256 x 64 x 64]
        # x_s[2]: [batch_size x 512 x 32 x 32]
        # x_s[3]: [batch_size x 1024 x 16 x 16]
        # x_s[4]: [batch_size x 2048 x 8 x 8]: will not be used because it's downpooled result
        
        #enc_out = self.pretrained_resnet50(img)
        return x, x_s

class Decoder(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv = ConvBlock(n_out*2, n_out)
        
    def forward(self, x, e):
        # up-conv 2x2
        x = self.deconv(x)
        # copy and crop
        #print(x.size(), e.size())
        x = torch.cat([x,e], axis=1)
        # conv 3x3, ReLU
        x = self.conv(x)
        return x

class PretrainedResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        
        # encoder
        self.enc = ResEncoder()
        """
        self.enc1 = Encoder(3, 64)
        self.enc2 = Encoder(64, 128)
        self.enc3 = Encoder(128, 256)
        self.enc4 = Encoder(256, 512)
        """
        
        # bottleneck
        #self.conv = ConvBlock(512, 1024)
        self.conv = ConvBlock(2048, 2048)
        
        # decoder
        self.dec1 = Decoder(2048, 1024)
        self.dec2 = Decoder(1024, 512)
        self.dec3 = Decoder(512, 256)
        self.dec4 = Decoder(256, 64)
        """
        self.dec1 = Decoder(1024, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)
        """
        
        # classifier
        self.deconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, dilation=1)
        self.outputs = nn.Conv2d(in_channels=32, out_channels=self.n_class, kernel_size=1, padding=0)
        
    def forward(self, x):
        
        # encoding (save the output before the pooling for later copy)
        """
        x1,p1 = self.enc1(x)
        x2,p2 = self.enc2(p1)
        x3,p3 = self.enc3(p2)
        x4,p4 = self.enc4(p3)
        """
        x, x_s = self.enc(x)
        
        # bottleneck
        b = self.conv(x)
        
        # decoding
        x = self.dec1(b, x_s[-2])
        x = self.dec2(x, x_s[-3])
        x = self.dec3(x, x_s[-4])
        x = self.dec4(x, x_s[-5])
        
        # classify
        x = self.deconv(x)
        outputs = self.outputs(x)
        
        return outputs   # size=(N, n_class, H, W)