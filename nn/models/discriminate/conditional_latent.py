#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import FullyConnectedLayer
from ai_old.util.params import init_params
from ai_old.util.factory import build_model_from_exp
import copy


@persistence.persistent_class
class _CLatentDiscriminator(nn.Module):
    def __init__(self,
        z_dims=512,
        n_layers=8,
        condition_dims=1,
    ):
        super().__init__()
        n_main_layers = n_layers - 1
        n_net0_layers = n_main_layers // 2
        n_net1_layers = n_main_layers - n_net0_layers
        if n_layers == 8:
            assert n_net0_layers == 3 and n_net1_layers == 4

        self.net0 = nn.Sequential(*[
            FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            )
            for _ in range(n_net0_layers)
        ])

        self.net1 = nn.Sequential(*[
            FullyConnectedLayer(
                z_dims,
                z_dims,
                activation='lrelu',
            )
            for _ in range(n_net1_layers)
        ])

        self.to_logit = FullyConnectedLayer(
            z_dims,
            1,
            activation='linear',
        )

        self.condition_encoder = nn.Linear(
            condition_dims,
            z_dims,
        )

    def forward(self, latent, condition):
        # print(latent)
        # print(condition)
        # print('~')
        # cenc =
        x = self.net0(latent) + self.condition_encoder(condition)
        # x = cenc
        x = self.net1(x)
        return self.to_logit(x)


@persistence.persistent_class
class CLatentDiscrimWrapper(nn.Module):
    def __init__(self,
        cfg,
        ae_exp='rec/21/3',
        z_dims=512,
        n_layers=8,
        condition_dims=1,
    ):
        super().__init__()
        assert cfg.model.G.ae_exp == ae_exp

        ae_model, ae_cfg = build_model_from_exp(ae_exp, 'G')
        assert ae_model.g.z_dims == z_dims
        self.e = copy.deepcopy(ae_model.e)
        self.e.requires_grad_(False)
        del ae_model

        self.d = _CLatentDiscriminator(
            z_dims=z_dims,
            n_layers=n_layers,
            condition_dims=condition_dims,
        )
        self.d.apply(init_params())

    def forward(self, img, condition):
        latent = self.e(img)
        return self.d(latent, condition)

    def prep_for_train_phase(self):
        self.d.requires_grad_(True)
