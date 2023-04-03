#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.conv import ConvBlock
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.blocks.etc import Flatten
from external.sg2 import persistence
import numpy as np


class SimpleFeatMapToLatent(nn.Module):
    def __init__(self, input_res, input_nc, z_dims):
        super().__init__()
        self.z_dims = z_dims
        n_down = int(np.log2(input_res))
        modules = [
            nn.Conv2d(input_nc, z_dims, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for i in range(n_down - 1):
            modules += [
                nn.Conv2d(z_dims, z_dims, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = FullyConnectedLayer(z_dims, z_dims)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.z_dims)
        x = self.linear(x)
        return x


@persistence.persistent_class
class LearnedHybridFeatMapToLatent(nn.Module):
    def __init__(self, z_dims):
        super().__init__()

        self.fullk = FullKFeatMapToLatent(z_dims)
        self.fullk_weight = nn.Parameter(torch.tensor(1.))

        self.down = DownFeatMapToLatent(z_dims)
        self.down_weight = nn.Parameter(torch.tensor(1.))

        self.fc = FcFeatMapToLatent(z_dims)
        self.fc_weight = nn.Parameter(torch.tensor(1.))

        self.final = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

    def forward(self, x):
        x = self.fullk(x) * self.fullk_weight + \
            self.down(x) * self.down_weight + \
            self.fc(x) * self.fc_weight
        return self.final(x)


@persistence.persistent_class
class FullKFeatMapToLatent(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.z_dims = z_dims

        self.convs = nn.Sequential(
            ConvBlock(z_dims, z_dims, norm='batch', actv='mish'),
            ConvBlock(
                z_dims,
                z_dims,
                k=4,
                pad=0,
                norm='none',
                actv='none',
            ),
        )

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


@persistence.persistent_class
class DownFeatMapToLatent(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.z_dims = z_dims

        self.convs = nn.Sequential(
            ConvBlock(z_dims, z_dims, norm='batch', actv='mish'),
            ConvBlock(
                z_dims,
                z_dims,
                s=2,
                norm='batch',
                actv='mish',
            ),
            ConvBlock(
                z_dims,
                z_dims,
                s=2,
                norm='batch',
                actv='mish',
            ),
        )

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


@persistence.persistent_class
class FcFeatMapToLatent(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.z_dims = z_dims

        self.conv = ConvBlock(z_dims, z_dims, norm='none', actv='none')

        self.fc = nn.Sequential(
            Flatten(),
            FullyConnectedLayer(
                z_dims * 16,
                z_dims,
                activation='linear',
                lr_multiplier=1,
            ),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
