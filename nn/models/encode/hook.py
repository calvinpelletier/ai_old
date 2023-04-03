#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence


@persistence.persistent_class
class FeatMapToLatent(nn.Module):
    def __init__(self,
        imsize,
        nc_in,
        z_dims,
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        self.z_dims = z_dims

        convs = []
        n_down = log2_diff(input_imsize, 1)
        nc = nc_in
        for i in range(n_down):
            next_nc = min(z_dims, nc * 2)
            convs.append(ConvBlock(
                nc,
                next_nc,
                s=2,
                norm=norm,
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



class E4eEncoder(nn.Module):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        num_ws=None,
        nc_in=3,
        nc_base=64,
        n_layers_per_res=[2, 4, 8, 4, 2],
        n_fine_down=2,
        n_med_down=2,
        n_coarse_down=1,
        n_coarse_ws=3,
        n_med_ws=4,
        n_fine_ws=10,
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        assert imsize == 128 and smallest_imsize == 4
        self.num_ws = num_ws
        n_down = log2_diff(imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down
        assert max(nc) <= z_dims
        assert z_dims == nc[-1]
        assert nc[0] == nc_base
        assert len(n_layers_per_res) == n_fine_down + n_med_down + n_coarse_down
        assert self.num_ws == n_coarse_ws + n_med_ws + \
            n_fine_ws
        assert self.num_ws == 3 * n_down + 2

        self.deepen = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            actv=actv,
        )
        fine_blocks = []
        med_blocks = []
        coarse_blocks = []
        for i in range(n_down):
            block = FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                actv=actv,
            )
            if i < n_fine_down:
                fine_blocks.append(block)
            elif i < n_fine_down + n_med_down:
                med_blocks.append(block)
            else:
                coarse_blocks.append(block)
        self.fine = nn.Sequential(*fine_blocks)
        self.med = nn.Sequential(*med_blocks)
        self.coarse = nn.Sequential(*coarse_blocks)

        fmtls = []
        for i in range(self.num_ws):
            if i < n_coarse_ws:
                fmtl = FeatMapToLatent(
                    imsize=smallest_imsize,
                    nc_in=nc[-1],
                    z_dims=z_dims,
                    norm=norm,
                    actv=actv,
                )
            elif i < n_coarse_ws + n_med_ws:
                fmtl = FeatMapToLatent(
                    imsize=smallest_imsize * 2 ** n_coarse_down,
                    nc_in=nc[-(1 + n_coarse_down)],
                    z_dims=z_dims,
                    norm=norm,
                    actv=actv,
                )
            else:
                fmtl = FeatMapToLatent(
                    imsize=smallest_imsize * 2 ** (n_coarse_down + n_med_down),
                    nc_in=nc[-(1 + n_coarse_down + n_med_down)],
                    z_dims=z_dims,
                    norm=norm,
                    actv=actv,
                )
            fmtls.append(fmtl)
        self.fmtls = nn.ModuleList(fmtls)

        self.laylayer1 = nn.Conv2d(
            nc[-(1 + n_coarse_down)],
            512,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.latlayer2 = nn.Conv2d(
            nc[-(1 + n_coarse_down + n_med_down)],
            512,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        deepened = self.deepen(x)

        fine = self.fine(deepened)
        med = self.med(fine)
        coarse = self.coarse(med)

        med_contextualized = self.laylayer1(med) + coarse

        w0 = self.fmtls[0](coarse)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)

        for i in range(1, self.num_ws):
            self.

        features = coarse
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.n_coarse_ws:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.n_coarse_ws + self.n_med_ws:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w
