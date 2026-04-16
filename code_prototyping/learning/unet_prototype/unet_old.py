import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    # def forward(self, x):
    #     e1 = self.enc1(x)
    #     e2 = self.enc2(self.pool(e1))
    #     e3 = self.enc3(self.pool(e2))
    #     e4 = self.enc4(self.pool(e3))
    #     b = self.bottleneck(self.pool(e4))
    #     d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
    #     d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
    #     d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
    #     d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
    #     return torch.sigmoid(self.conv_last(d1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        if d4.size() != e4.size():
            d4 = F.interpolate(d4, size=e4.shape[2:])
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.conv_last(d1))