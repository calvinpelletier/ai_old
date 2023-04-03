#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.linear import LinearBlock
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import VectorUpConvBlock, SimpleUpConvBlock, ConvToImg
from ai_old.nn.blocks.mod import SleModulation


# class Encoder(Unit):
#     def __init__(self,)

class GeneratorWrapper(Unit):
    def __init__(self,
        z_dims=256,

        n_mlp=4,
        f_norm='batch',
        f_weight_norm=False,
        f_actv='mish',

        g_type='light',
        nc_base=32,
        g_norm='batch',
        g_weight_norm=False,
        g_actv='glu',
    ):
        super().__init__()
        self.z_dims = z_dims

        f_layers = []
        for i in range(n_mlp):
            f_layers.append(LinearBlock(
                z_dims,
                z_dims,
                norm=f_norm,
                weight_norm=f_weight_norm,
                actv=f_actv,
            ))
        self.f = nn.Sequential(*f_layers)

        if g_type == 'light':
            self.g = Generator(
                z_dims=z_dims,
                nc_base=nc_base,
                norm=g_norm,
                weight_norm=g_weight_norm,
                actv=g_actv,
            )
        # elif g_type == 'style':
        #     self.g = StyleGenerator(
        #         z_dims=z_dims,
        #         nc_base=nc_base,
        #         norm=g_norm,
        #         weight_norm=g_weight_norm,
        #         actv=g_actv,
        #     )
        else:
            raise ValueError(f'{g_type}')

    def forward(self, batch_size):
        z_entangled = torch.randn(batch_size, self.z_dims, device='cuda')
        z_entangled = F.normalize(z_entangled, dim=1)
        z = self.f(z_entangled)
        img = self.g(z)
        return img


class Generator(Unit):
    def __init__(self,
        smallest_imsize=4,
        z_dims=256,
        nc_base=32,
        norm='batch',
        weight_norm=False,
        actv='glu',
    ):
        super().__init__()

        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(6)][::-1]

        self.z_to_img = VectorUpConvBlock(
            z_dims,
            nc[0],
            k=smallest_imsize,
            norm=norm,
            actv='glu',
        )

        # self.initial_conv = nn.Conv2d(nc[0], nc[0], 3, padding=1)

        # main path through generator
        self.up0 = SimpleUpConvBlock(
            nc[0],
            nc[1],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        self.up1 = SimpleUpConvBlock(
            nc[1],
            nc[2],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        self.up2 = SimpleUpConvBlock(
            nc[2],
            nc[3],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        self.up3 = SimpleUpConvBlock(
            nc[3],
            nc[4],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        self.up4 = SimpleUpConvBlock(
            nc[4],
            nc[5],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # shortcuts for strong gradient flow
        self.shortcut0 = SleModulation(nc[1], nc[4])
        self.shortcut1 = SleModulation(nc[2], nc[5])

        # to img
        self.conv_to_img = ConvToImg(nc[5])

    def forward(self, z):
        # initial z to img
        z = z.unsqueeze(dim=2)
        z = z.unsqueeze(dim=3)
        x4 = self.z_to_img(z)
        x4 = F.normalize(x4, dim=1)
        # x4 = self.initial_conv(x4)

        # upsampling layers
        x8 = self.up0(x4)
        x16 = self.up1(x8)
        x32 = self.up2(x16)
        x64 = self.shortcut0(self.up3(x32), x8)
        x128 = self.shortcut1(self.up4(x64), x16)

        # to img
        out = self.conv_to_img(x128)
        return out
