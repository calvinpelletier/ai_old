#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.blocks.linear import LinearBlock
from ai_old.nn.blocks.etc import CoralLayer


class Mlp(nn.Module):
    def __init__(self,
        dims_in,
        hidden,
        dims_out,
        norm='batch',
        weight_norm=False,
        actv='mish',
        coral_output=False,
    ):
        super().__init__()
        layers = []
        prev = dims_in
        for neurons in hidden:
            layers.append(LinearBlock(
                prev,
                neurons,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            prev = neurons

        if coral_output:
            layers.append(CoralLayer(prev, dims_out))
        else:
            layers.append(nn.Linear(prev, dims_out))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
