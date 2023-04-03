#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import NoiseResUpConvBlock
from ai_old.nn.blocks.conv import ConvToImg
from ai_old.nn.blocks.quant import SimpleNoiseResUpConvBlock


class NoiseResDecoder(nn.Module):
    def __init__(self,
        imsize=256,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        norm='batch',
        actv='mish',
        conv_clamp=None,
        onnx=False,
        simple=False,
    ):
        super().__init__()
        self.n_blocks = log2_diff(imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(self.n_blocks + 1)]
        nc = nc[::-1]

        for i in range(self.n_blocks):
            if simple:
                block = SimpleNoiseResUpConvBlock(
                    smallest_imsize * (2 ** (i + 1)),
                    nc[i],
                    nc[i+1],
                )
            else:
                block = NoiseResUpConvBlock(
                    smallest_imsize * (2 ** (i + 1)),
                    nc[i],
                    nc[i+1],
                    norm=norm,
                    actv=actv,
                    conv_clamp=conv_clamp,
                    onnx=onnx,
                )
            setattr(self, f'b{i}', block)

        self.final = ConvToImg(nc[-1])

    def forward(self, x, noise_mode='random'):
        for i in range(self.n_blocks):
            x = getattr(self, f'b{i}')(x, noise_mode=noise_mode)
        return self.final(x)

    def fuse_model(self):
        for i in range(self.n_blocks):
            getattr(self, f'b{i}').fuse_model()
