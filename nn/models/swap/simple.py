#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import FullyConnectedLayer
from ai_old.util.factory import build_model_from_exp
import copy
from ai_old.util.params import init_params


@persistence.persistent_class
class _LatentSwapperV0(nn.Module):
    def __init__(self,
        z_dims=512,
    ):
        super().__init__()

        disentangler_layers = []
        for _ in range(4):
            disentangler_layers.append(FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            ))
        self.disentangler = nn.Sequential(*disentangler_layers)

        entangler_layers = []
        for _ in range(4):
            entangler_layers.append(FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            ))
        self.entangler = nn.Sequential(*entangler_layers)

        self.gvec = nn.Parameter(torch.randn([z_dims]))

    def forward(self, latent, src_gender, magnitude=1.):
        # disentangle
        disentangled = self.disentangler(latent)

        # interpolate
        gvec_mult = src_gender * 2. - 1. # 0/1 to -1/1
        # print('gvec_mult', gvec_mult, gvec_mult.shape)
        gvec = gvec_mult * self.gvec
        new_disentangled = disentangled + gvec

        # entangle
        new_latent = self.entangler(new_disentangled)
        return new_latent


@persistence.persistent_class
class _LatentSwapperV1(nn.Module):
    def __init__(self,
        z_dims=512,
    ):
        super().__init__()

        net0_layers = []
        for _ in range(4):
            net0_layers.append(FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            ))
        self.net0 = nn.Sequential(*net0_layers)

        net1_layers = []
        for _ in range(4):
            net1_layers.append(FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            ))
        self.net1 = nn.Sequential(*net1_layers)

        self.gvec = nn.Parameter(torch.randn([z_dims]))

    def forward(self, latent, src_gender, magnitude=1.):
        # net0
        net0_out = self.net0(latent)

        # interpolate
        gvec_mult = src_gender * 2. - 1. # 0/1 to -1/1
        gvec = gvec_mult * self.gvec * magnitude
        interpolated_net0_out = net0_out + gvec

        # net1
        net1_out = self.net1(interpolated_net0_out)

        new_latent = latent + net1_out
        return new_latent


@persistence.persistent_class
class SimpleSwapper(nn.Module):
    def __init__(self,
        cfg,
        ae_exp='rec/21/3',
        z_dims=512,
        version='v0',
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.z_dims = z_dims

        ae_model, ae_cfg = build_model_from_exp(ae_exp, 'G')
        assert ae_model.imsize == self.imsize
        assert ae_model.intermediate == 'zspace'
        assert ae_model.g.z_dims == z_dims
        self.g = copy.deepcopy(ae_model.g)
        self.g.requires_grad_(False)
        self.e = copy.deepcopy(ae_model.e)
        self.e.requires_grad_(False)
        del ae_model

        if version == 'v0':
            self.t = _LatentSwapperV0(z_dims=z_dims)
        elif version == 'v1':
            self.t = _LatentSwapperV1(z_dims=z_dims)
        else:
            raise Exception(version)
        self.t.apply(init_params())

    def forward(self, img, gender, magnitude=1.):
        # encode
        latent = self.e(img)

        # swap
        swap_latent = self.t(latent, gender, magnitude)

        # generate
        swap_img = self.g(swap_latent)
        return swap_img

    def prep_for_train_phase(self):
        self.t.requires_grad_(True)
