#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock, ClampResDownConvBlock, \
    ResDownConvBlock
from ai_old.nn.blocks.conv import ConvBlock, CustomConvBlock
from ai_old.nn.blocks.encode import FeatMapToLatentViaFc


class Encoder(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        n_layers_per_res=[2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
        conv_clamp=None,
        dropout_after_squeeze_layers=False,
    ):
        super().__init__()

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        assert len(n_layers_per_res) == n_down_up
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

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
        for i in range(n_down_up):
            if n_layers_per_res[i] == 0:
                if dropout_after_squeeze_layers and n_layers_per_res[i-1] != 0:
                    blocks.append(nn.Dropout(0.1))
                if conv_clamp is not None:
                    down = ClampResDownConvBlock(
                        nc[i],
                        nc[i+1],
                        k_down=3,
                        norm=norm,
                        weight_norm=False,
                        actv=actv,
                        conv_clamp=conv_clamp,
                    )
                else:
                    down = ResDownConvBlock(
                        nc[i],
                        nc[i+1],
                        k_down=3,
                        norm=norm,
                        weight_norm=False,
                        actv=actv,
                    )
            else:
                down = FancyMultiLayerDownBlock(
                    nc[i],
                    nc[i+1],
                    n_layers=n_layers_per_res[i],
                    norm=norm,
                    actv=actv,
                    conv_clamp=conv_clamp,
                )
            blocks.append(down)
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


@persistence.persistent_class
class ZEncoder(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        z_dims=512,
        num_ws=None,
        nc_in=3,
        nc_base=32,
        n_layers_per_res=[2, 4, 8, 4, 2],
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.num_ws = num_ws

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down_up
        assert max(nc) <= z_dims
        assert z_dims == nc[-1]
        assert nc[0] == nc_base
        if input_imsize == 128 and smallest_imsize == 4:
            assert n_down_up == 5
            if nc_base == 32 and z_dims == 512:
                assert nc == [32, 64, 128, 256, 512, 512]

        # main blocks
        blocks = [ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]
        for i in range(n_down_up):
            down = FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            )
            blocks.append(down)
        self.main = nn.Sequential(*blocks)

        # to z
        self.to_z = FeatMapToLatentViaFc(
            smallest_imsize,
            smallest_imsize,
            z_dims,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, input):
        x = self.main(input)
        z = self.to_z(x)
        if self.num_ws is None:
            return z
        else:
            ws = z.unsqueeze(1).repeat([1, self.num_ws, 1])
            return ws
