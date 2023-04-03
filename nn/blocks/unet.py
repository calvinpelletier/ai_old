#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.conv import DoubleConvBlock, ConvBlock


class DownUnetBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        self.shortcut = ConvBlock(
            nc1,
            nc2,
            k=k_shortcut,
            s=2,
            norm=norm,
            weight_norm=weight_norm,
            actv='none',
        )

        self.main = DoubleConvBlock(
            nc1,
            nc2,
            nc2,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.down = nn.Conv2d(nc2, nc2, 3, padding=1, stride=2)

    def forward(self, x):
        unet_res = self.main(x)
        out = self.down(unet_res)
        out = out + self.shortcut(x)
        return out, unet_res


class UpUnetBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        self.shortcut = ConvBlock(
            nc1,
            nc2,
            k=k_shortcut,
            norm=norm,
            weight_norm=weight_norm,
            actv='none',
        )

        self.main = DoubleConvBlock(
            nc1 * 2,
            nc1,
            nc2,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

    def forward(self, input, unet_res):
        upsampled = self.up(input)
        concat = torch.cat((upsampled, unet_res), dim=1)
        out = self.main(concat)
        out = out + self.shortcut(upsampled)
        return out
