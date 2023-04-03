#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.params import init_params
from ai_old.nn.models.encode.e4e import E4eEncoder, E4eEncoderNoProg, E4eEncoderW0Only
from ai_old.util.factory import build_model_from_exp
import copy
import os
import ai_old.constants as c
from external.sg2.unit import SynthesisNetwork
from ai_old.util.etc import log2_diff, AttrDict
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.models.decode.style import LearnedConstStyleDecoder, ZOnlyStyleDecoder
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.models.encode.fm2l import LearnedHybridFeatMapToLatent
import math
from ai_old.util.pretrained import build_pretrained_e4e
from ai_old.nn.models.facegen.student import Sg2Student


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        n_layers_per_res=[2, 2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down_up
        assert nc[0] == nc_base
        if input_imsize == 128 and smallest_imsize == 4:
            assert n_down_up == 5
            if nc_base == 64 and nc_max == 512:
                assert nc == [64, 128, 256, 512, 512, 512]

        blocks = [ConvBlock(
            nc_in,
            nc[0],
            norm='none',
            weight_norm=False,
            actv=actv,
        )]
        for i in range(n_down_up):
            down = FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                weight_norm=False,
                actv=actv,
            )
            blocks.append(down)
        self.net = nn.Sequential(*blocks)

        # to z
        self.to_z = LearnedHybridFeatMapToLatent(z_dims)

        self.w_count = 2 * int(math.log(input_imsize, 2)) - 2

    def forward(self, input):
        enc = self.net(input)
        z = self.to_z(enc)
        ws = z.repeat(self.w_count, 1, 1).permute(1, 0, 2)
        return ws


@persistence.persistent_class
class E4e(nn.Module):
    def __init__(self,
        cfg,
        e_type='reg',
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.intermediate = 'modspace'

        if e_type == 'reg':
            self.e = E4eEncoder(self.imsize)
        elif e_type == 'no_prog':
            self.e = E4eEncoderNoProg(self.imsize)
        elif e_type == 'w0_only':
            self.e = E4eEncoderW0Only(self.imsize)
        else:
            raise Exception(e_type)
        self.e.apply(init_params())
        # self.e.load_state_dict(torch.load(os.path.join(
        #     c.PRETRAINED_MODELS,
        #     'arcface/model_ir_se50.pth',
        # )), strict=False)
        # self.e = Encoder(
        #     input_imsize=self.imsize,
        # )
        # self.e.apply(init_params())

        self.g = SynthesisNetwork(
            w_dim=512,
            img_resolution=self.imsize,
            img_channels=3,
            channel_base=64 * self.imsize,
            channel_max=512,
            num_fp16_res=4,
            conv_clamp=256,
            fp16_channels_last=False,
        )
        # self.g.apply(init_params())

    def forward(self, x):
        ws = self.e(x)
        return self.g(ws)

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.g.requires_grad_(True)


class E4eInitialized(nn.Module):
    def __init__(self,
        cfg,
        g_exp='distill/1/0',
        e_initialized=True,
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.intermediate = 'modspace'

        if g_exp is not None:
            g = build_model_from_exp(g_exp, 'G_ema', return_cfg=False)
            self.g = copy.deepcopy(g)
            del g
        else:
            self.g = Sg2Student(
                AttrDict({'dataset': AttrDict({'imsize': self.imsize})}),
                nc_base=16,
            )
        self.g.train()
        self.g.requires_grad_(True)

        self.e = build_pretrained_e4e(eval=False, load_weights=e_initialized)
        self.e.train()
        self.e.requires_grad_(True)

    def forward(self, x):
        ws = self.e(x)
        return self.g(ws)

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.g.requires_grad_(True)
