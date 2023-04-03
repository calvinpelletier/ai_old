#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.conv import ConvBlock
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.blocks.etc import Flatten
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock, ResDownConvBlock
from ai_old.util.etc import log2_diff
import numpy as np
from external.sg2 import persistence


class FancyFeatMapToLatent(nn.Module):
    def __init__(self,
        nc_in,
        z_dims,
        n_layers=2,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.enc = FancyMultiLayerDownBlock(
            nc_in,
            z_dims,
            n_layers=2,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        self.linear = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, (2, 3))
        return self.linear(x)


class LightFeatMapToLatent(nn.Module):
    def __init__(self,
        nc_in,
        z_dims,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.enc = ResDownConvBlock(
            nc_in,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        self.linear = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, (2, 3))
        return self.linear(x)


@persistence.persistent_class
class FeatMapToLatentViaFc(nn.Module):
    def __init__(self,
        input_imsize,
        smallest_imsize,
        nc_in,
        z_dims,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        convs = []
        if input_imsize == smallest_imsize:
            nc = nc_in
        else:
            n_down = log2_diff(input_imsize, smallest_imsize)
            nc = nc_in
            for i in range(n_down):
                next_nc = min(z_dims, nc * 2)
                convs.append(ConvBlock(
                    nc,
                    next_nc,
                    s=2,
                    norm=norm,
                    weight_norm=weight_norm,
                    actv=actv,
                ))
                nc = next_nc

        convs.append(ConvBlock(
            nc,
            z_dims,
            norm='none',
            weight_norm=False,
            actv='none',
        ))
        self.convs = nn.Sequential(*convs)

        self.fc = nn.Sequential(
            Flatten(),
            FullyConnectedLayer(
                z_dims * smallest_imsize**2,
                z_dims,
                activation='linear',
                lr_multiplier=1,
            ),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x)
        return x


@persistence.persistent_class
class FeatMapToLatent(nn.Module):
    def __init__(self,
        input_imsize,
        smallest_imsize,
        nc_in,
        z_dims,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.z_dims = z_dims

        convs = []
        if input_imsize == smallest_imsize:
            nc = nc_in
        else:
            n_down = log2_diff(input_imsize, smallest_imsize)
            nc = nc_in
            for i in range(n_down):
                next_nc = min(z_dims, nc * 2)
                convs.append(ConvBlock(
                    nc,
                    next_nc,
                    s=2,
                    norm=norm,
                    weight_norm=weight_norm,
                    actv=actv,
                ))
                nc = next_nc

        convs.append(nn.Conv2d(
            nc,
            z_dims,
            kernel_size=smallest_imsize,
            padding=0,
        ))
        self.convs = nn.Sequential(*convs)

        self.linear = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.z_dims)
        x = self.linear(x)
        return x



class FullDownAndFlatten(nn.Module):
    def __init__(self,
        imsize,
        c1,
        c2,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.c2 = c2
        convs = []
        for i in range(int(np.log2(imsize))):
            convs.append(ConvBlock(
                c1 if i == 0 else c2,
                c2,
                s=2,
                norm=norm,
                weight_norm=weight_norm,
                actv=activation,
            ))
        self.convs = nn.Sequential(*convs)
        self.linear = FullyConnectedLayer(
            c2,
            c2,
            activation='linear',
            lr_multiplier=1,
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.c2)
        x = self.linear(x)
        return x
