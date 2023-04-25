
import torch
import torch.nn as nn

class Custom_FCN1(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv1_4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(64)
        self.max1 = nn.MaxPool2d((2,2)) # stride is default equal to kernel size, i.e. (2,2)
        
        self.conv2_1 = nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2_2 = nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_3 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv2_4 = nn.Conv2d(64, 16, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv2_5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(128)
        self.max2 = nn.MaxPool2d((2,2))
        
        self.conv3_1 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv3_2 = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3_3 = nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv3_4 = nn.Conv2d(128, 32, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv3_5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(256)
        self.max3 = nn.MaxPool2d((2,2))
        
        self.conv4_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv4_2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv4_4 = nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv4_5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(512)
        self.max4 = nn.MaxPool2d((2,2))
        
        self.conv5_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 128, kernel_size=5, stride=1, padding=2, dilation=1)
        self.conv5_4 = nn.Conv2d(512, 128, kernel_size=7, stride=1, padding=3, dilation=1)
        self.conv5_5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)
        self.max5 = nn.MaxPool2d((2,2))
        
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
        x1 = self.relu(self.conv1_1(x))
        x1 = self.relu(self.conv1_2(x1))
        x1 = self.relu(self.conv1_3(x1))
        x1 = self.relu(self.conv1_4(x1))
        x1 = self.max1(self.bnd1(x1))
                         
                         
        x2_1 = self.relu(self.conv2_1(x1))
        x2_2 = self.relu(self.conv2_2(x1))
        x2_3 = self.relu(self.conv2_3(x1))
        x2_4 = self.relu(self.conv2_4(x1))
        x2_5 = self.relu(self.conv2_5(torch.cat((x2_1, x2_2, x2_3, x2_4), dim=1)))
        x2 = self.max2(self.bnd2(x2_5))
                         
                         
        x3_1 = self.relu(self.conv3_1(x2))
        x3_2 = self.relu(self.conv3_2(x2))
        x3_3 = self.relu(self.conv3_3(x2))
        x3_4 = self.relu(self.conv3_4(x2))
        x3_5 = self.relu(self.conv3_5(torch.cat((x3_1, x3_2, x3_3, x3_4), dim=1)))
        x3 = self.max3(self.bnd3(x3_5))
                         
                         
        x4_1 = self.relu(self.conv4_1(x3))
        x4_2 = self.relu(self.conv4_2(x3))
        x4_3 = self.relu(self.conv4_3(x3))
        x4_4 = self.relu(self.conv4_4(x3))
        x4_5 = self.relu(self.conv4_5(torch.cat((x4_1, x4_2, x4_3, x4_4), dim=1)))
        x4 = self.max4(self.bnd4(x4_5))
                         
                         
        x5_1 = self.relu(self.conv5_1(x4))
        x5_2 = self.relu(self.conv5_2(x4))
        x5_3 = self.relu(self.conv5_3(x4))
        x5_4 = self.relu(self.conv5_4(x4))
        x5_5 = self.relu(self.conv5_5(torch.cat((x5_1, x5_2, x5_3, x5_4), dim=1)))
        x5 = self.max5(self.bnd5(x5_5))
        

        # DECODER
        y1 = self.bn1(self.relu(self.deconv1(x5)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))

        # We don't use softmax since nn.CrossEntropyLoss applies Softmax
        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)