#!/usr/bin/env python3
from ai_old.nn.models import Unit
from ai_old.nn.blocks.sg2 import StyleVectorizer


class Disentangler(Unit):
    def __init__(self,
        z_dims=512,
        n_layers=8,
        lr_mul=0.1,
    ):
        super().__init__()
        self.net = StyleVectorizer(
            z_dims,
            n_layers,
            lr_mul=lr_mul,
        )

    def forward(self, seed):
        return self.net(seed)
