import torch.nn as nn
from torchvision import models
from torch import reshape

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, inp):
        return inp
    
class Resnet_FCN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_resnet34 = models.resnet34(pretrained = True)
        self.pretrained_resnet34.avgpool = Identity()
        self.pretrained_resnet34.fc = Identity()
        for param in self.pretrained_resnet34.parameters():
            param.requires_grad = False
        
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # ENCODER
        x5 = self.pretrained_resnet34(x)
        
        # for some reason x5 comes out as (16, 25088) so it needs to be reshaped
        x5 = reshape(x5, (-1, 512, 7, 7))

        # DECODER
        '''
        y1 = self.bn1(self.relu(self.deconv1(x5)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))
        '''
        
        y1 = self.relu(self.bn1(self.deconv1(x5)))
        y2 = self.relu(self.bn2(self.deconv2(y1)))
        y3 = self.relu(self.bn3(self.deconv3(y2)))
        y4 = self.relu(self.bn4(self.deconv4(y3)))
        y5 = self.relu(self.bn5(self.deconv5(y4)))

        # We don't use softmax since nn.CrossEntropyLoss applies Softmax
        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)
    
    def display_params(self):
        for name, param in self.pretrained_resnet34.named_parameters():
            print(name, param.requires_grad)