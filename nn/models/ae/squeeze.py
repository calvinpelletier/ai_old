#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import ResUpConvBlock, FancyMultiLayerDownBlock
from ai_old.nn.blocks.conv import ConvToImg, ConvBlock
from ai_old.util.params import init_params


@persistence.persistent_class
class SqueezeAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        n_layers_per_res=[2, 2, 4, 8, 4, 2],
        norm='batch',
        weight_norm=False,
        enc_actv='mish',
        dec_actv='mish',
    ):
        super().__init__()
        self.intermediate = 'enc'
        self.imsize = cfg.dataset.imsize
        n_down_up = log2_diff(self.imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]
        assert len(n_layers_per_res) == n_down_up

        # outer blocks
        enc_blocks = [ConvBlock(
            nc_in,
            nc[0],
            norm='none',
            weight_norm=False,
            actv=enc_actv,
        )]
        dec_blocks = [ConvToImg(nc[0])]

        # inner blocks
        for i in range(n_down_up):
            enc_blocks.append(FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                actv=enc_actv,
            ))
            dec_blocks.append(ResUpConvBlock(
                nc[i+1],
                nc[i],
                norm=norm,
                weight_norm=weight_norm,
                actv=dec_actv,
            ))

        self.e = nn.Sequential(*enc_blocks)
        self.g = nn.Sequential(*dec_blocks[::-1])

        self.apply(init_params())

    def forward(self, x):
        encoding = self.e(x)
        return self.g(encoding)

    def prep_for_train_phase(self):
        self.requires_grad_(True)
