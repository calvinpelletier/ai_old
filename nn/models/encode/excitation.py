#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.mod import DownDoubleExcitationBlock
from ai_old.util.etc import log2_diff


class ExcitationModulatedEncoder(Unit):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        z_dims=512,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert max(nc) <= nc_max
        assert nc[0] == nc_base

        # initial deepen
        self.initial = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # down blocks
        down_blocks = []
        for i in range(n_down_up):
            down_blocks.append(DownDoubleExcitationBlock(
                nc[i],
                nc[i+1],
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.enc = nn.ModuleList(down_blocks)

    def forward(self, x, z):
        x = self.initial(x)
        for enc in self.enc:
            x = enc(x, z)
        return x
