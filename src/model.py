import torch

from torch import nn


class ResBlock(nn.Module):
    def __init__(self, c, ker=3, pad=1):
        super().__init__()
        self.l1 = nn.Conv2d(
            c, c,
            kernel_size=ker,
            padding=pad
        )
        self.l2 = nn.Conv2d(
            c, c,
            kernel_size=ker,
            padding=pad
        )
        self.act = nn.GELU()

    def forward(self, x):
        out = self.act(self.l1(x))
        out = self.l2(out)
        out += x
        return out


class ResModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            ResBlock(128),
            nn.Conv2d(128, 32, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            ResBlock(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            ResBlock(16),
            nn.BatchNorm2d(16)
        )

        self.dec = nn.Sequential(
            ResBlock(16),
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GELU(),
            ResBlock(32),
            nn.ConvTranspose2d(
                32, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.GELU(),
            ResBlock(128),
            nn.ConvTranspose2d(
                128, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, b=None):
        x = self.enc(x)
        if self.training and b:
            n = torch.rand(x.shape) * x.max() / (2**(b+1))
            x = x + n
        x = self.dec(x)
        return x
