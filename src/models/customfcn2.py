
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bnd = nn.BatchNorm2d(n_out)
        self.pool = nn.MaxPool2d((2,2), return_indices=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.pool(self.bnd(self.relu(self.conv(x))))        

class Decoder(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(n_in, n_out, kernel_size=3, stride=1, padding=1, dilation=1, output_padding=0)
        self.bnd = nn.BatchNorm2d(n_out)
        self.unpool = nn.MaxUnpool2d((2,2))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, p):
        return self.unpool(self.bnd(self.relu(self.deconv(x))), p)
        
class Custom_FCN2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.enc1 = Encoder(3,32)
        self.enc2 = Encoder(32,64)
        self.enc3 = Encoder(64,128)
        self.enc4 = Encoder(128,256)
        self.enc5 = Encoder(256,512)

        self.mconv = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1)
        self.bnd = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        
        self.dec1 = Decoder(512, 512)
        self.dec2 = Decoder(512, 256)
        self.dec3 = Decoder(256, 128)
        self.dec4 = Decoder(128, 64)
        self.dec5 = Decoder(64, 32)
        
        self.outconv = nn.Conv2d(32, 32, kernel_size=1)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # ENCODER
        x1,p1 = self.enc1(x)
        x2,p2 = self.enc2(x1)
        x3,p3 = self.enc3(x2)
        x4,p4 = self.enc4(x3)
        x5,p5 = self.enc5(x4)
        
        b = self.bnd(self.relu(self.mconv(x5)))
        
        #print(x.size(), x1.size(), x2.size(), x3.size(), x4.size(), x5.size(), b.size(), p5.size())

        # DECODER
        y1 = self.dec1(b,p5)
        y2 = self.dec2(y1,p4)
        y3 = self.dec3(y2,p3)
        y4 = self.dec4(y3,p2)
        y5 = self.dec5(y4,p1)

        out = self.relu(self.outconv(y5))
        # We don't use softmax since nn.CrossEntropyLoss applies Softmax
        score = self.classifier(out)

        #print(y1.size(), y2.size(), y3.size(), y4.size(), y5.size(), out.size(), score.size())

        return score  # size=(N, n_class, H, W)
