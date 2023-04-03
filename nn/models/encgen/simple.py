#!/usr/bin/env python3
from ai_old.nn.models import Unit
from ai_old.nn.models.encode.simple import SimpleZEncoder
from ai_old.nn.models.facegen.light import ExtraLightFaceGenerator
from ai_old.nn.models.facegen.style import MinimalStyleGenerator
from ai_old.util.factory import build_model_from_exp
from ai_old.util.params import requires_grad


class SimpleStyleRecOnlyEncGen(Unit):
    def __init__(self,
        # shared
        imsize=128,
        smallest_imsize=4,
        z_dims=512,

        # encoder
        k_init=5,
        nc_in=3,
        e_nc_base=32,
        im2vec='kernel',
        e_norm='batch',
        e_weight_norm=False,
        e_actv='mish',

        # generator
        g_nc_base=32,
    ):
        super().__init__()

        self.enc = SimpleZEncoder(
            input_imsize=imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            k_init=k_init,
            nc_in=nc_in,
            nc_base=e_nc_base,
            im2vec=im2vec,
            norm=e_norm,
            weight_norm=e_weight_norm,
            actv=e_actv,
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


class SimpleRecOnlyEncGen(Unit):
    def __init__(self,
        # shared
        imsize=128,
        smallest_imsize=4,
        z_dims=512,

        # encoder
        k_init=5,
        nc_in=3,
        e_nc_base=32,
        im2vec='kernel',
        e_norm='batch',
        e_weight_norm=False,
        e_actv='mish',

        # generator
        g_norm='batch',
        g_weight_norm=False,
        g_actv='glu',
        g_init_from_exp=None,
        g_unfreeze_at_epoch=None,
    ):
        super().__init__()
        self.g_init_from_exp = g_init_from_exp
        self.g_unfreeze_at_epoch = g_unfreeze_at_epoch
        self.g_frozen = g_unfreeze_at_epoch is not None

        self.enc = SimpleZEncoder(
            input_imsize=imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            k_init=k_init,
            nc_in=nc_in,
            nc_base=e_nc_base,
            im2vec=im2vec,
            norm=e_norm,
            weight_norm=e_weight_norm,
            actv=e_actv,
        )

        self.gen = ExtraLightFaceGenerator(
            imsize=imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            norm=g_norm,
            weight_norm=g_weight_norm,
            actv=g_actv,
        )

    def forward(self, x):
        z = self.enc(x)
        img = self.gen(z)
        return img

    def init_params(self):
        self.enc.init_params()

        if self.g_init_from_exp is not None:
            self.gen = build_model_from_exp(
                self.g_init_from_exp,
                freeze=self.g_frozen,
                verbose=False,
            ).to('cuda')
        else:
            self.gen.init_params()

    def print_info(self):
        self.enc.print_info()
        self.gen.print_info()

    def end_of_epoch(self, epoch):
        if self.g_frozen and self.g_unfreeze_at_epoch is not None and \
                epoch >= self.g_unfreeze_at_epoch:
            self.g_frozen = False
            requires_grad(self.gen, True)
