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
class PspEncoder(nn.Module):
    def __init__(self,
        w_avg,
        imsize=128,
        smallest_imsize=8,
        z_dims=512,
        num_ws=12,
        nc_in=3,
        nc_base=64,
        n_layers_per_res=[2, 4, 8, 4],
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        self.num_ws = num_ws
        n_down = log2_diff(imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down
        assert max(nc) <= z_dims
        assert z_dims == nc[-1]
        assert nc[0] == nc_base

        self.register_buffer('avg_ws', w_avg.unsqueeze(dim=1).repeat(
            1, self.num_ws, 1, 1))

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
        for i in range(self.num_ws):
            fmtls.append(FeatMapToLatent(
                imsize=smallest_imsize,
                nc_in=nc[-1],
                z_dims=z_dims,
                actv=actv,
            ))
        self.fmtls = nn.ModuleList(fmtls)

    def forward(self, x, base_ws):
        if base_ws is None:
            base_ws = self.avg_ws
        x = self.deepen(x)
        x = self.main(x)
        ws = base_ws.clone().detach()
        for i, fmtl in enumerate(self.fmtls):
            ws[:, i, :, :] += fmtl(x)
        return ws

    def get_avg_ws(self):
        return avg_ws


@persistence.persistent_class
class RestylePspEncoder(nn.Module):
    def __init__(self,
        avg_ws,
        imsize=128,
        smallest_imsize=8,
        z_dims=512,
        num_ws=12,
        nc_in=6,
        nc_base=64,
        n_layers_per_res=[2, 4, 8, 4],
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        self.num_ws = num_ws
        n_down = log2_diff(imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down
        assert max(nc) <= z_dims
        assert z_dims == nc[-1]
        assert nc[0] == nc_base

        self.register_buffer('avg_ws', avg_ws)
        self.register_buffer('avg_img', torch.zeros((1, 3, imsize, imsize)))

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
        for i in range(self.num_ws):
            fmtls.append(FeatMapToLatent(
                imsize=smallest_imsize,
                nc_in=nc[-1],
                z_dims=z_dims,
                actv=actv,
            ))
        self.fmtls = nn.ModuleList(fmtls)

    def set_avg_img(self, avg_img):
        self.register_buffer('avg_img', avg_img)

    def forward(self, x, base_img, base_ws):
        if base_img is None:
            base_img = self.avg_img.repeat(x.shape[0], 1, 1, 1)
        else:
            base_img = base_img.clone().detach().requires_grad_(True)

        if base_ws is None:
            base_ws = self.avg_ws.repeat(x.shape[0], 1, 1)
        else:
            base_ws = base_ws.clone().detach().requires_grad_(True)

        x = self.deepen(torch.cat([base_img, x], dim=1))
        x = self.main(x)
        ws = base_ws.clone()
        for i, fmtl in enumerate(self.fmtls):
            ws[:, i, :] += fmtl(x)
        return ws
