#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import FullyConnectedLayer
from ai_old.util.params import init_params
from ai_old.util.factory import build_model_from_exp
import copy


@persistence.persistent_class
class _LatentBinaryClassifier(nn.Module):
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
        return self.net(latent).squeeze()


@persistence.persistent_class
class BinaryClassiferFromPretrainedEncoder(nn.Module):
    def __init__(self,
        cfg,
        ae_exp='rec/21/3',
        z_dims=512,
        n_layers=8,
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.z_dims = z_dims

        ae_model, ae_cfg = build_model_from_exp(ae_exp, 'G')
        assert ae_model.imsize == self.imsize
        assert ae_model.intermediate == 'zspace'
        assert ae_model.g.z_dims == z_dims
        self.e = copy.deepcopy(ae_model.e)
        self.e.requires_grad_(False)
        del ae_model

        self.c = _LatentBinaryClassifier(
            z_dims=z_dims,
            n_layers=n_layers,
        )
        self.c.apply(init_params())

    def forward(self, img):
        enc = self.e(img)
        pred = self.c(enc)
        return pred

    def prep_for_train_phase(self):
        self.c.requires_grad_(True)
