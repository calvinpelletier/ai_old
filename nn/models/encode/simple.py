#!/usr/bin/env python3
import torch.nn as nn
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.conv import ConvBlock, ConvToImg, CustomConvBlock
from ai_old.nn.blocks.res import ResDownConvBlock, ClampResDownConvBlock
from ai_old.nn.blocks.encode import FeatMapToLatent, FeatMapToLatentViaFc


class SimpleZEncoder(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        z_dims=256,
        k=3,
        k_init=5,
        nc_in=3,
        nc_base=16,
        im2vec='kernel',
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        n_down = log2_diff(input_imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * 2**i) for i in range(n_down + 1)]

        # deepen
        layers = [ConvBlock(
            nc_in,
            nc[0],
            k=k_init,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]

        # downsample
        for i in range(n_down):
            layers.append(ResDownConvBlock(
                nc[i],
                nc[i+1],
                k_down=k,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))

        # final
        if im2vec == 'kernel':
            im2vec_module = FeatMapToLatent
        elif im2vec == 'mlp':
            im2vec_module = FeatMapToLatentViaFc
        else:
            raise ValueError(im2vec)

        layers.append(im2vec_module(
            smallest_imsize,
            4,
            nc[-1],
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        ))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class SimpleEncoder(nn.Module):
    def __init__(self,
        input_imsize,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        norm='batch',
        actv='mish',
        conv_clamp=None,
    ):
        super().__init__()
        n_down = log2_diff(input_imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * 2**i) for i in range(n_down + 1)]

        # initial block
        if conv_clamp is not None:
            blocks = [CustomConvBlock(
                nc_in,
                nc[0],
                norm='none',
                actv=actv,
                conv_clamp=conv_clamp,
            )]
        else:
            blocks = [ConvBlock(
                nc_in,
                nc[0],
                norm='none',
                actv=actv,
            )]

        # main blocks
        for i in range(n_down):
            if conv_clamp is not None:
                blocks.append(ClampResDownConvBlock(
                    nc[i],
                    nc[i+1],
                    k_down=3,
                    norm=norm,
                    weight_norm=False,
                    actv=actv,
                    conv_clamp=conv_clamp,
                ))
            else:
                blocks.append(ResDownConvBlock(
                    nc[i],
                    nc[i+1],
                    k_down=3,
                    norm=norm,
                    weight_norm=False,
                    actv=actv,
                ))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)
