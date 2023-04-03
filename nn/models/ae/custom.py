#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import ResUpConvBlock, ResDownConvBlock
from ai_old.nn.blocks.conv import ConvToImg, ConvBlock
from ai_old.util.params import init_params
from ai_old.nn.models.encode.squeeze import ZEncoder
from ai_old.nn.models.encode.style import StyleEncoder
from ai_old.nn.models.decode.style import StyleDecoder
from ai_old.util.factory import build_model_from_exp
import copy


@persistence.persistent_class
class PriModEncoders(nn.Module):
    def __init__(self, e_pri, e_mod):
        super().__init__()
        self.e_pri = e_pri
        self.e_mod = e_mod

    def forward(self, x):
        z = self.e_pri(x)
        encoding = self.e_mod(x, z)
        return encoding, z





@persistence.persistent_class
class CustomizableAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        z_dims=512,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        norm='batch',
        weight_norm=False,
        e_pri_exp=None,
        e_pri_frozen=False,
        e_pri_actv='swish',
        e_pri_layers_per_res=[2, 4, 8, 4, 2],
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.e_pri_frozen = e_pri_frozen
        self.intermediate = 'enc_plus_z'

        if e_pri_exp is None:
            e_pri = ZEncoder(
                input_imsize=self.imsize,
                smallest_imsize=4,
                z_dims=z_dims,
                num_ws=None,
                nc_in=nc_in,
                nc_base=nc_base,
                n_layers_per_res=e_pri_layers_per_res,
                norm=norm,
                weight_norm=weight_norm,
                actv=e_pri_actv,
            )
            e_pri.apply(init_params())
        else:
            og_model, og_cfg = build_model_from_exp(e_pri_exp, 'G')
            e_pri = copy.deepcopy(og_model.e)
            e_pri.num_ws = None
            e_pri.requires_grad_(not e_pri_frozen)
            del og_model

        e_mod = StyleEncoder(
            imsize=self.imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
            to_z=False,
        )
        e_mod.apply(init_params())

        self.e = PriModEncoders(e_pri, e_mod)

        self.g = StyleDecoder(
            imsize=self.imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
        )
        self.g.apply(init_params())

    def forward(self, x):
        encoding, z = self.e(x)
        return self.g(encoding, z)

    def prep_for_train_phase(self):
        if not self.e_pri_frozen:
            self.e.e_pri.requires_grad_(True)
        self.e.e_mod.requires_grad_(True)
        self.g.requires_grad_(True)
