import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class dubleconv(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.block(x)


class encoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()

        self.conv = dubleconv(input_channels,output_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool

class decoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
    
        self.up = nn.ConvTranspose2d(input_channels,output_channels,kernel_size=2,stride=2)
        self.conv = dubleconv(input_channels,output_channels)

    def forward(self,x,skip):
        up = self.up(x)
        stacked = torch.cat([skip,up],dim=1)
        conv = self.conv(stacked)
        return conv

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = encoder(1,64)
        self.enc2 = encoder(64,128)
        self.enc3 = encoder(128,256)
        self.enc4 = encoder(256,512)

        self.bottleneck = dubleconv(512,1024)

        self.dec1 = decoder(1024,512)
        self.dec2 = decoder(512,256)
        self.dec3 = decoder(256,128)
        self.dec4 = decoder(128,64)
    
        self.fin_conv = nn.Conv2d(64,1,kernel_size = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        s1,x = self.enc1(x)
        s2,x = self.enc2(x)
        s3,x = self.enc3(x)
        s4,x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec1(x,s4)
        x = self.dec2(x,s3)
        x = self.dec3(x,s2)
        x = self.dec4(x,s1)

        x = self.fin_conv(x)
        x = self.sigmoid(x)
        return x

def dice_loss(pred,target,smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    overlap = (pred*target).sum()
    dice = (2. * overlap + smooth) / (pred.sum() + target.sum() + smooth)
    return 1-dice

#Testing
if __name__ == "__main__":
    model = Unet()
    x = torch.randn(1, 1, 256, 256)
    block = dubleconv(1,64)
    y = block(x)
    print(y.shape)
    y_pred = torch.tensor([[0.8, 0.1], [0.4, 0.9]])
    y_true = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)

    loss = dice_loss(y_pred, y_true)
    print(loss)

