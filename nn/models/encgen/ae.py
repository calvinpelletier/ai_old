#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisNetwork
from ai_old.nn.models.encode.squeeze import ZEncoder
from ai_old.util import config
import copy
from ai_old.util.params import init_params
from ai_old.util.factory import build_model_from_exp
from ai_old.nn.models.encode.style import StyleEncoderToModspace
from ai_old.nn.models.encode.arcface import ArcFaceWrapper


@persistence.persistent_class
class PriModEncoders(nn.Module):
    def __init__(self,
        e_pri,
        imsize=128,
        z_dims=512,
        e_mod_type='style',
    ):
        super().__init__()

        self.e_pri = e_pri

        assert e_mod_type == 'style'
        self.e_mod = StyleEncoderToModspace(input_imsize=imsize)
        self.e_mod.apply(init_params())

    def forward(self, x):
        z = self.e_pri(x)
        ws = self.e_mod(x, z)
        return ws


@persistence.persistent_class
class ModspaceAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        zspace_ae_exp='rec/12/2',
        reset_e_pri=False,
        freeze_e_pri=False,
        freeze_g=True,
        e_mod_type='style',
    ):
        super().__init__()
        self.freeze_e_pri = freeze_e_pri
        self.freeze_g = freeze_g

        if reset_e_pri:
            assert not self.freeze_e_pri

        # transfer modules from the original generator
        og_G, og_cfg = build_model_from_exp(zspace_ae_exp, 'G')
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        self.num_ws = og_G.num_ws
        self.z_dims = og_G.z_dims
        self.imsize = og_G.imsize
        e_pri = copy.deepcopy(og_G.e)
        del og_G

        # optionally train e_pri from scratch
        # TODO: rebuild e_pri
        if reset_e_pri:
            e_pri.apply(init_params())

        # build encoder
        self.e = PriModEncoders(
            e_pri=e_pri,
            imsize=self.imsize,
            z_dims=self.z_dims,
            e_mod_type=e_mod_type,
        )

    def forward(self, input):
        ws = self.e(input)
        output = self.g(ws)
        return output

    def prep_for_train_phase(self):
        if not self.freeze_g:
            self.g.requires_grad_(True)
        if not self.freeze_e_pri:
            self.e.e_pri.requires_grad_(True)
        self.e.e_mod.requires_grad_(True)


@persistence.persistent_class
class GeneratorInitializedAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        g_exp='facegen/8/1',
        e_type='slow_squeeze_excite',
        e_nc_base=32,
        e_n_layers_per_res=[2, 4, 8, 4, 2],
        e_norm='batch',
        e_weight_norm=False,
        e_actv='mish',
    ):
        super().__init__()
        self.intermediate = 'modspace'

        # load the original generator and its config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G')

        # transfer modules from the original generator
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        self.num_ws = og_G.f.num_ws
        self.z_dims = og_G.z_dims
        self.imsize = og_G.g.img_resolution
        del og_G

        # build encoder
        if e_type == 'slow_squeeze_excite':
            self.e = ZEncoder(
                input_imsize=self.imsize,
                smallest_imsize=4,
                z_dims=self.z_dims,
                num_ws=self.num_ws,
                nc_in=3,
                nc_base=e_nc_base,
                n_layers_per_res=e_n_layers_per_res,
                norm=e_norm,
                weight_norm=e_weight_norm,
                actv=e_actv,
            )
            self.e.apply(init_params())
        elif e_type == 'initialized_arcface':
            self.e = ArcFaceWrapper(
                pretrained=True,
                frozen=False,
                num_ws=self.num_ws,
            )
        else:
            raise Exception(f'{e_type}')

    def forward(self, input):
        ws = self.e(input)
        output = self.g(ws)
        return output

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)


# @persistence.persistent_class
# class OldGeneratorInitializedAutoencoder(nn.Module):
#     def __init__(self,
#         cfg,
#         exp='invert/0/1',
#     ):
#         super().__init__()
#
#         # load the original generator and its config
#         og_G, og_cfg = build_model_from_exp(exp, 'G')
#
#         # transfer modules from the original generator
#         self.g = copy.deepcopy(og_G.g)
#         self.g.requires_grad_(False)
#         self.e = copy.deepcopy(og_G.e)
#         self.e.requires_grad_(False)
#         self.num_ws = og_G.f.num_ws
#         self.z_dims = og_G.z_dims
#         self.imsize = og_G.imsize
#         del og_G
#
#     def forward(self, input):
#         w = self.e(input)
#         ws = w.unsqueeze(1).repeat([1, self.num_ws, 1])
#         output = self.g(ws)
#         return output
#
#     def prep_for_train_phase(self):
#         self.e.requires_grad_(True)
