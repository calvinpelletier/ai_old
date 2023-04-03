#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import normalize_2nd_moment, FullyConnectedLayer
from ai_old.util.factory import build_model_from_exp
import copy
from ai_old.util.params import init_params
import external.sg2.misc as misc


@persistence.persistent_class
class SeedToLatent(nn.Module):
    def __init__(self,
        z_dims=512,
        n_layers=8,
    ):
        super().__init__()
        self.z_dims = z_dims

        layers = []
        for _ in range(n_layers):
            layers.append(FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, seed):
        misc.assert_shape(seed, [None, self.z_dims])
        x = normalize_2nd_moment(seed.to(torch.float32))
        out = self.net(x)
        return out


@persistence.persistent_class
class LatentGenerator(nn.Module):
    def __init__(self,
        cfg,
        ae_exp='rec/21/3',
        z_dims=512,
        n_layers=8,
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.z_dims = z_dims
        assert cfg.model.D.ae_exp == ae_exp

        ae_model, ae_cfg = build_model_from_exp(ae_exp, 'G')
        assert ae_model.imsize == self.imsize
        assert ae_model.intermediate == 'zspace'
        assert ae_model.g.z_dims == z_dims
        self.g = copy.deepcopy(ae_model.g)
        self.g.requires_grad_(False)
        del ae_model

        self.f = SeedToLatent(
            z_dims=z_dims,
            n_layers=n_layers,
        )
        self.f.apply(init_params())

    def forward(self, seed):
        latent = self.f(seed)
        img = self.g(latent)
        return img

    def prep_for_train_phase(self):
        self.f.requires_grad_(True)
