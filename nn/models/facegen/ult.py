#!/usr/bin/env python3
import torch.nn.functional as F
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import VectorUpConvBlock
from ai_old.util.etc import log2_diff
from ai_old.nn.models.facegen.adalin import AdalinModulatedGenerator
from ai_old.nn.models.facegen.excitation import ExcitationModulatedGenerator
from ai_old.nn.models.transform.disentangle import Disentangler


class GenOnlyUltModel(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
        nc_max=512,

        # f
        f='mlp',
        f_n_layers=6,
        f_lr_mul=0.1,

        # g
        g='excitation',
        g_actv='mish',
    ):
        super().__init__()
        n_down_up = log2_diff(imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]
        nc = nc[::-1]

        assert f == 'mlp'
        self.f = Disentangler(
            z_dims=z_dims,
            n_layers=f_n_layers,
            lr_mul=f_lr_mul,
        )

        if g == 'adalin':
            g_cls = AdalinModulatedGenerator
        elif g == 'excitation':
            g_cls = ExcitationModulatedGenerator
        else:
            raise ValueError(g)
        self.g = g_cls(
            output_imsize=imsize,
            init_imsize=smallest_imsize,
            nc_in=3,
            nc_base=nc_base,
            nc_max=nc_max,
            z_dims=z_dims,
            norm='batch',
            weight_norm=False,
            actv=g_actv,
        )

        self.z_to_img = VectorUpConvBlock(
            z_dims,
            nc[0],
            k=smallest_imsize,
            norm='batch',
            actv='glu',
        )

    def forward(self, seed):
        # disentangle z
        z = self.f(seed)

        # z to initial img
        initial = self.z_to_img(z.unsqueeze(dim=2).unsqueeze(dim=3))
        initial = F.normalize(initial, dim=1)

        # generate
        return self.g(initial, z)
