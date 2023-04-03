#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import FullyConnectedLayer
from ai_old.util.params import init_params
from ai_old.util.factory import build_model_from_exp
import copy


@persistence.persistent_class
class _LatentDiscriminator(nn.Module):
    def __init__(self,
        z_dims=512,
        n_layers=8,
    ):
        super().__init__()

        layers = []
        for _ in range(n_layers - 1):
            layers.append(FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            ))
        layers.append(FullyConnectedLayer(
            z_dims,
            1,
            activation='linear',
        ))
        self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)


@persistence.persistent_class
class LatentDiscriminatorWrapper(nn.Module):
    def __init__(self,
        cfg,
        ae_exp='rec/21/3',
        z_dims=512,
        n_layers=8,
    ):
        super().__init__()
        assert cfg.model.G.ae_exp == ae_exp

        ae_model, ae_cfg = build_model_from_exp(ae_exp, 'G')
        assert ae_model.g.z_dims == z_dims
        self.e = copy.deepcopy(ae_model.e)
        self.e.requires_grad_(False)
        del ae_model

        self.d = _LatentDiscriminator(z_dims=z_dims, n_layers=n_layers)
        self.d.apply(init_params())

    def forward(self, img):
        latent = self.e(img)
        return self.d(latent)

    def prep_for_train_phase(self):
        self.d.requires_grad_(True)
