import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        return skip, self.pool(skip)

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        return self.conv(torch.cat([skip, self.up(x)], dim=1))

class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # Encoder: 256 → 128 → 64 → 32
        self.enc1 = Encoder(in_channels, 16)
        self.enc2 = Encoder(16, 32)
        self.enc3 = Encoder(32, 64)

        # Bottleneck bei 32×32
        self.bottleneck = DoubleConv(64, 128)

        # Decoder: 32 → 64 → 128 → 256
        self.dec1 = Decoder(128, 64)
        self.dec2 = Decoder(64,  32)
        self.dec3 = Decoder(32,  16)

        self.fin_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        s1, x = self.enc1(x)   # skip: 256×256, 16ch
        s2, x = self.enc2(x)   # skip: 128×128, 32ch
        s3, x = self.enc3(x)   # skip:  64×64,  64ch

        x = self.bottleneck(x) # 32×32, 128ch

        x = self.dec1(x, s3)   # → 64×64,  64ch
        x = self.dec2(x, s2)   # → 128×128, 32ch
        x = self.dec3(x, s1)   # → 256×256, 16ch

        return self.fin_conv(x)

def dice_loss(pred,target,smooth=1e-6):
    target = target.to(pred.device).float()
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    overlap = (pred*target).sum()
    dice = (2. * overlap + smooth) / (pred.sum() + target.sum() + smooth)
    return 1-dice

if __name__ == "__main__":
    model = Unet()
    x = torch.randn(1, 1, 256, 256)
    print(f"Output: {model(x).shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {total:,}")