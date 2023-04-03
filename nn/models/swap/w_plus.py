#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import ai_old.constants as c
from external.sg2 import persistence
from external.sg2.unit import FullyConnectedLayer
from ai_old.util.params import init_params
from external.e4e.models.stylegan2.model import Generator
from ai_old.util.factory import legacy_build_model_from_exp, build_model_from_exp
from ai_old.util.pretrained import build_pretrained_sg2


@persistence.persistent_class
class _GlobalLatentSwapper(nn.Module):
    def __init__(self,
        z_dims=512,
    ):
        super().__init__()
        self.gvec = nn.Parameter(torch.randn([z_dims]))

    def forward(self, latent, src_gender, magnitude=1.):
        gvec_mult = src_gender * 2. - 1. # 0/1 to -1/1
        return gvec_mult * self.gvec * magnitude


@persistence.persistent_class
class PerLatentSwapper(nn.Module):
    def __init__(self,
        z_dims=512,
        n_latents=18,
    ):
        super().__init__()
        self.n_latents = n_latents

        swappers = []
        for i in range(self.n_latents):
            swapper = _GlobalLatentSwapper(z_dims=z_dims)
            swappers.append(swapper)
        self.swappers = nn.ModuleList(swappers)

    def forward(self, w_plus, gender, magnitude=1.):
        delta = []
        for i in range(self.n_latents):
            d = self.swappers[i](w_plus[:, i, :], gender, magnitude)
            delta.append(d.unsqueeze(dim=1))
        delta = torch.cat(delta, dim=1)
        return w_plus + delta, delta


@persistence.persistent_class
class PretrainedSynthSwapper(nn.Module):
    def __init__(self, n_latents=18):
        super().__init__()
        self.n_latents = n_latents

        self.model = legacy_build_model_from_exp(
            'gender-lerp/1/3',
            verbose=False,
        ).to('cuda')
        self.model.eval()

    def forward(self, w_plus, gender, magnitude=1.):
        w0 = w_plus[:, 0, :]
        new_w0 = self.model(w0, gender)['z2']
        delta = (new_w0 - w0).unsqueeze(dim=1).repeat(1, self.n_latents, 1)
        delta *= magnitude
        return w_plus + delta


@persistence.persistent_class
class SwapperAndGenerator(nn.Module):
    def __init__(self,
        cfg,
        swapper_type='per_latent',
    ):
        super().__init__()
        self.swapper_type = swapper_type

        G = build_pretrained_sg2()
        self.g = G.synthesis.to('cuda').eval()
        self.g.requires_grad_(False)

        if swapper_type == 'per_latent':
            self.f = PerLatentSwapper()
            self.f.apply(init_params())
        elif swapper_type == 'pretrained_ss':
            self.f = PretrainedSynthSwapper()
            self.f.requires_grad_(False)
        else:
            raise Exception(swapper_type)

    def forward(self, w_plus, gender, magnitude=1.):
        w_plus_swap, delta = self.f(w_plus, gender, magnitude)
        swap = self.g(w_plus_swap)
        return swap

    def prep_for_train_phase(self):
        if self.swapper_type != 'pretrained_ss':
            self.f.requires_grad_(True)


class SwapperAndPtiGenerator(nn.Module):
    def __init__(self, pti_g_path, swap_exp=None):
        super().__init__()

        G = build_pretrained_sg2(path_override=pti_g_path)
        self.g = G.synthesis
        self.g.requires_grad_(False)

        if swap_exp is None:
            self.f = PretrainedSynthSwapper()
        else:
            self.f = build_model_from_exp(swap_exp, 'G', return_cfg=False).f
        self.f.requires_grad_(False)

    def forward(self, w_plus, gender, magnitude=1.):
        w_plus_swap = self.f(w_plus, gender, magnitude)
        swap = self.g(w_plus_swap, noise_mode='const')
        return swap


























# asdf
