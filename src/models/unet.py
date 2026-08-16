import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Double conv with instance norm + LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Bilinear upsample + conv (avoids checkerboard artefacts vs transposed conv)."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if input size is odd
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SEMUNet(nn.Module):
    """
    Lightweight 3-level U-Net for SEM image denoising.

    Architecture summary:
        Encoder:  1 → 32 → 64 → 128
        Bottleneck: 128 → 256
        Decoder:  256+128 → 128 → 128+64 → 64 → 64+32 → 32
        Output:   32 → 1 (sigmoid)

    Parameters: ~540k (extremely lightweight for fast CPU inference)
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, base_ch)          # 256x256 → 256x256
        self.enc2 = ConvBlock(base_ch, base_ch*2)  # 128x128 → 128x128
        self.enc3 = ConvBlock(base_ch*2, base_ch*4) # 64x64 → 64x64
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch*4, base_ch*8)  # 32x32

        # Decoder
        self.dec3 = UpBlock(base_ch*8, base_ch*4, base_ch*4)
        self.dec2 = UpBlock(base_ch*4, base_ch*2, base_ch*2)
        self.dec1 = UpBlock(base_ch*2, base_ch,   base_ch)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)

