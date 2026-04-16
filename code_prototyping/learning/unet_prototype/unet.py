import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # -------- Encoder (ResNet) --------
        self.enc0 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            resnet.bn1,
            resnet.relu,
        )

        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1      # 64
        self.enc2 = resnet.layer2      # 128
        self.enc3 = resnet.layer3      # 256
        self.enc4 = resnet.layer4      # 512

        # -------- Bottleneck --------
        self.bottleneck = ConvBlock(512, 1024)

        # -------- Decoder --------
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(512 + 256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(256 + 128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(128 + 64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # -------- Encoder --------
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # -------- Bottleneck --------
        b = self.bottleneck(e4)

        # -------- Decoder (mit sicherem Resize) --------
        d4 = self.up4(b)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e0.shape[2:]:
            d1 = F.interpolate(d1, size=e0.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))

        out = self.conv_last(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(out)


if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 513, 513)
    y = model(x)
    print(y.shape)
