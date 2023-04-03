#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.conv import ConvBlock
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock


@persistence.persistent_class
class FeatMapToLatent(nn.Module):
    def __init__(self,
        imsize,
        nc_in,
        z_dims,
        actv='mish',
    ):
        super().__init__()
        self.z_dims = z_dims

        convs = []
        n_down = log2_diff(imsize, 1)
        nc = nc_in
        for i in range(n_down):
            next_nc = min(z_dims, nc * 2)
            convs.append(ConvBlock(
                nc,
                next_nc,
                s=2,
                norm='none',
                actv=actv,
            ))
            nc = next_nc
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


@persistence.persistent_class
class LightE4eEncoder(nn.Module):
    def __init__(self,
        imsize=256,
        n_styles=18,
        smallest_imsize=8,
        z_dims=512,
        nc_in=3,
        nc_base=32,
        n_layers_per_res=[2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        self.style_count = n_styles

        n_down = log2_diff(imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down + 1)]
        assert len(n_layers_per_res) == n_down

        self.deepen = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            actv=actv,
        )
        blocks = []
        for i in range(n_down):
            blocks.append(FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                actv=actv,
            ))
        self.main = nn.Sequential(*blocks)

        fmtls = []
        for i in range(self.style_count):
            fmtls.append(FeatMapToLatent(
                imsize=smallest_imsize,
                nc_in=nc[-1],
                z_dims=z_dims,
                actv=actv,
            ))
        self.fmtls = nn.ModuleList(fmtls)

        self.progressive_stage = self.style_count

    def get_deltas_starting_dimensions(self):
        return list(range(self.style_count))

    def set_progressive_stage(self, new_stage):
        self.progressive_stage = new_stage
        print(f'changed progressive stage to: {new_stage}')

    def forward(self, x):
        x = self.deepen(x)
        x = self.main(x)

        w0 = self.fmtls[0](x)
        w = w0.unsqueeze(1).repeat(1, self.style_count, 1)

        stage = self.progressive_stage
        for i in range(1, min(stage + 1, self.style_count)):
            w[:, i, :] += self.fmtls[i](x)

        return w
