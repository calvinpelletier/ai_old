#!/usr/bin/env python3
from ai_old.nn.models import Unit
from ai_old.nn.models.encode.uenet import LightUenet
from ai_old.nn.models.facegen.style import StyleGenerator, DynamicInitStyleGenerator


class LightUenetRecOnlyEncGen(Unit):
    def __init__(self,
        # shared
        imsize=128,
        smallest_imsize=4,
        z_dims=512,

        # encoder
        e_nc_base=32,

        # generator
        g_nc_base=32,
    ):
        super().__init__()

        self.enc = LightUenet(
            input_imsize=imsize,
            smallest_imsize=smallest_imsize,
            nc_in=3,
            nc_base=e_nc_base,
            z_dims=z_dims,
            g_dynamic_init=False,
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        self.gen = StyleGenerator(
            imsize=imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            nc_base=g_nc_base,
        )

    def forward(self, x):
        zz = self.enc(x)
        img = self.gen(zz)
        return img

    def init_params(self):
        self.enc.init_params()
        self.gen.init_params()

    def print_info(self):
        self.enc.print_info()
        self.gen.print_info()


class LightUenetDynamicInitRecOnlyEncGen(Unit):
    def __init__(self,
        # shared
        imsize=128,
        smallest_imsize=4,
        z_dims=512,

        # encoder
        e_nc_base=32,

        # generator
        g_nc_base=32,
    ):
        super().__init__()

        self.enc = LightUenet(
            input_imsize=imsize,
            smallest_imsize=smallest_imsize,
            nc_in=3,
            nc_base=e_nc_base,
            z_dims=z_dims,
            g_dynamic_init=True,
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        self.gen = DynamicInitStyleGenerator(
            imsize=imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            nc_base=g_nc_base,
        )

    def forward(self, x):
        zz, initial = self.enc(x)
        img = self.gen(zz, initial)
        return img

    def init_params(self):
        self.enc.init_params()
        self.gen.init_params()

    def print_info(self):
        self.enc.print_info()
        self.gen.print_info()
