"""Lightweight U-Net model for binary road segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two Conv2D + BatchNorm + ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small U-Net that outputs one-channel logits."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(base_channels * 2, base_channels)

        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _match_size(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Resize decoder features if rounding creates a one-pixel mismatch."""
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.up4(bottleneck)
        dec4 = self._match_size(dec4, enc4)
        dec4 = self.decoder4(torch.cat([dec4, enc4], dim=1))

        dec3 = self.up3(dec4)
        dec3 = self._match_size(dec3, enc3)
        dec3 = self.decoder3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.up2(dec3)
        dec2 = self._match_size(dec2, enc2)
        dec2 = self.decoder2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.up1(dec2)
        dec1 = self._match_size(dec1, enc1)
        dec1 = self.decoder1(torch.cat([dec1, enc1], dim=1))

        return self.output_conv(dec1)


def main() -> None:
    model = UNet(base_channels=32)
    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_output = model(dummy_input)
    print(f"Input shape: {tuple(dummy_input.shape)}")
    print(f"Output logits shape: {tuple(dummy_output.shape)}")


if __name__ == "__main__":
    main()
