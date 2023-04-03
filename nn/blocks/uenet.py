#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.adalin import Adalin
from ai_old.nn.blocks.etc import get_actv
from ai_old.nn.blocks.conv import ConvBlock, DoubleConvBlock
from ai_old.nn.blocks.encode import FancyFeatMapToLatent, LightFeatMapToLatent


class AdalinUpUenetBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        to_mod_layers=2,
        k=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
        pad='auto',
        use_bias=False,
    ):
        super().__init__()

        if pad == 'auto':
            pad = (k - 1) // 2

        self.pad1 = nn.ReflectionPad2d(pad)
        self.conv1 = nn.Conv2d(
            nc1 * 2, nc1,
            kernel_size=k,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        self.norm1 = Adalin(nc1, z_dims)
        self.actv1 = get_actv(actv)

        self.pad2 = nn.ReflectionPad2d(pad)
        self.conv2 = nn.Conv2d(
            nc1, nc2,
            kernel_size=k,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        self.norm2 = Adalin(nc2, z_dims)
        self.actv2 = get_actv(actv)

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

        self.shortcut = ConvBlock(
            nc1,
            nc2,
            k=k,
            norm=norm,
            weight_norm=weight_norm,
            actv='none',
        )

        self.to_mod = FancyFeatMapToLatent(
            nc2,
            z_dims,
            n_layers=to_mod_layers,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, x, z, unet_res):
        upsampled = self.up(x)
        concat = torch.cat((upsampled, unet_res), dim=1)

        out = self.pad1(concat)
        out = self.conv1(out)
        out = self.norm1(out, z)
        out = self.actv1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, z)
        out = self.actv2(out)

        main_out = out + self.shortcut(upsampled)
        mod = self.to_mod(out)

        return main_out, mod


class SimpleUpUenetBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        self.shortcut = ConvBlock(
            nc1,
            nc2,
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

        self.to_mod = LightFeatMapToLatent(
            nc2,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, x, z, unet_res):
        upsampled = self.up(x)
        concat = torch.cat((upsampled, unet_res), dim=1)
        main_out = self.main(concat)
        mod = self.to_mod(main_out)
        out = main_out + self.shortcut(upsampled)
        return out, mod
