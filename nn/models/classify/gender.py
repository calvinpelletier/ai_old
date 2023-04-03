#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.util.params import init_params
from ai_old.nn.models.encode.arcface import ArcFaceWrapper
from ai_old.nn.blocks.mlp import Mlp


class MinimalZGenderClassifier(Unit):
    def __init__(self, z_dims=512):
        super().__init__()
        self.net = nn.Linear(z_dims, 1)

    def forward(self, z):
        return self.net(z).squeeze()


class ImgGenderClassifier(Unit):
    def __init__(self,
        hidden=[512, 256, 128, 32, 8],
        norm='batch',
        weight_norm=True,
        actv='mish',
        unfreeze_arcface_at_epoch=4,
    ):
        super().__init__()

        # encoder
        self.e = ArcFaceWrapper(unfreeze_at_epoch=unfreeze_arcface_at_epoch)

        # classifier
        self.c = Mlp(
            self.e.dims_out,
            hidden,
            1, # n classes
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, x):
        feat = self.e(x)
        out = self.c(feat).squeeze()
        return {'gender_pred': out}

    def init_params(self):
        print('running custom param init')
        self.e.init_params()
        self.c.apply(init_params())

    def end_of_epoch(self, epoch):
        self.e.end_of_epoch(epoch)
