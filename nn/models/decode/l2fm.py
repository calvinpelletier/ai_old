#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.conv import ConvBlock, UpConvBlock, VectorUpConvBlock
from external.sg2.unit import FullyConnectedLayer
from external.sg2 import persistence


@persistence.persistent_class
class LearnedHybridLatentToFeatMap(nn.Module):
    def __init__(self, z_dims):
        super().__init__()

        self.fullk = FullKLatentToFeatMap(z_dims)
        self.fullk_weight = nn.Parameter(torch.tensor(1.))

        self.up = UpLatentToFeatMap(z_dims)
        self.up_weight = nn.Parameter(torch.tensor(1.))

        # self.fc = FcLatentToFeatMap(z_dims)
        # self.fc_weight = nn.Parameter(torch.tensor(1.))

        self.final = ConvBlock(
            z_dims,
            z_dims,
            norm='none',
            actv='none',
        )

    def forward(self, x):
        x = self.fullk(x) * self.fullk_weight + \
            self.up(x) * self.up_weight
        return self.final(x)


@persistence.persistent_class
class FullKLatentToFeatMap(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.z_dims = z_dims

        self.linear = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

        self.convs = nn.Sequential(
            VectorUpConvBlock(
                z_dims,
                z_dims,
                k=4,
                norm='batch',
                actv='glu',
            ),
            ConvBlock(z_dims, z_dims, norm='batch', actv='swish'),
        )


    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(dim=2)
        x = x.unsqueeze(dim=3)
        x = self.convs(x)
        return x


@persistence.persistent_class
class UpLatentToFeatMap(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.z_dims = z_dims

        self.linear = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

        self.convs = nn.Sequential(
            UpConvBlock(
                z_dims,
                z_dims,
                norm='batch',
            ),
            UpConvBlock(
                z_dims,
                z_dims,
                norm='batch',
            ),
            ConvBlock(z_dims, z_dims, norm='batch', actv='swish'),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(dim=2)
        x = x.unsqueeze(dim=3)
        x = self.convs(x)
        return x
