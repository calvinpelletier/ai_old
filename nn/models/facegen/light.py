#!/usr/bin/env python3
import torch.nn.functional as F
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import VectorUpConvBlock, SimpleUpConvBlock, ConvToImg
from ai_old.nn.blocks.mod import SleModulation
from ai_old.util.etc import log2_diff


class ExtraLightFaceGenerator(Unit):
    def __init__(self,
        imsize=128,
        z_dims=512,
        smallest_imsize=4,
        shortcuts_from=[1, 2], # layer idxs
        shortcut_dist=3, # n layers
        #   4    8    16   32   64   128
        nc=[512, 512, 256, 128, 64, 32],
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        assert len(nc) == log2_diff(imsize, smallest_imsize) + 1

        # hardcoding for now
        # TODO: make dynamic
        assert imsize == 128 and smallest_imsize == 4
        assert shortcuts_from == [1, 2] and shortcut_dist == 3

        self.z_to_img = VectorUpConvBlock(
            z_dims,
            nc[0],
            k=smallest_imsize,
            norm=norm,
            actv='glu',
        )

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

        # upsampling layers
        x8 = self.up0(x4)
        x16 = self.up1(x8)
        x32 = self.up2(x16)
        x64 = self.shortcut0(self.up3(x32), x8)
        x128 = self.shortcut1(self.up4(x64), x16)

        # to img
        out = self.conv_to_img(x128)
        return out
