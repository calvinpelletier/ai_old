#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.blocks.linear import LinearBlock
from ai_old.nn.blocks.mlp import Mlp
from ai_old.nn.models import Unit


"""
input: disentangled latent vector of dims (batch_size, 512)
output: dict of attribute predictions
"""
class ZAttrPredictor(Unit):
    def __init__(self,
        z_dims=512,
        n_shared_layers=2,
        attr_hidden_layers=[512, 128, 32, 8],
        norm='batch',
        weight_norm=True,
        actv='mish',
    ):
        super().__init__()

        shared_layers = []
        for _ in range(n_shared_layers):
            shared_layers.append(LinearBlock(
                z_dims,
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.shared = nn.Sequential(*shared_layers)

        self.mouth = self._build_attr_predictor(
            z_dims, attr_hidden_layers, norm, weight_norm, actv)
        self.glasses = self._build_attr_predictor(
            z_dims, attr_hidden_layers, norm, weight_norm, actv)

    def _build_attr_predictor(self,
            z_dims,
            hidden,
            norm,
            weight_norm,
            actv,
        ):
        return Mlp(
            z_dims,
            hidden,
            1, # dims_out
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, z):
        intermediate = self.shared(z)
        return {
            'pred_mouth': self.mouth(intermediate).squeeze(),
            'pred_glasses': self.glasses(intermediate).squeeze(),
        }
