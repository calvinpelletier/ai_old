#!/usr/bin/env python3
from ai_old.nn.models import Unit
from ai_old.nn.models.encode.arcface import ArcFaceWrapper
from ai_old.nn.models.facegen.style import MinimalStyleGenerator


class ArcfaceStyleRecOnlyEncGen(Unit):
    def __init__(self,
        # shared
        imsize=128,
        smallest_imsize=4,
        z_dims=512,

        # encoder
        e_pretrained=False,
        e_frozen=False,
        e_unfreeze_at_epoch=None,

        # generator
        g_nc_base=32,
    ):
        super().__init__()

        self.enc = ArcFaceWrapper(
            pretrained=e_pretrained,
            frozen=e_frozen,
            unfreeze_at_epoch=e_unfreeze_at_epoch,
        )

        self.gen = MinimalStyleGenerator(
            imsize=imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            nc_base=g_nc_base,
        )

    def forward(self, x):
        z = self.enc(x)
        img = self.gen(z)
        return img

    def init_params(self):
        self.enc.init_params()
        self.gen.init_params()

    def print_info(self):
        self.enc.print_info()
        self.gen.print_info()
