#!/usr/bin/env python3
import torch
from ai_old.nn.models import Unit
from ai_old.util.params import init_params
from ai_old.nn.models.encode.arcface import ArcFaceWrapper
from ai_old.nn.blocks.mlp import Mlp
from ai_old.util.age import get_max_age


class ImgAgeClassifier(Unit):
    def __init__(self,
        hidden=[512, 512, 512],
        norm='batch',
        weight_norm=True,
        actv='mish',
        unfreeze_arcface_at_epoch=4,
        scaled_age=False,
    ):
        super().__init__()

        # encoder
        self.e = ArcFaceWrapper(unfreeze_at_epoch=unfreeze_arcface_at_epoch)

        # classifier
        self.c = Mlp(
            self.e.dims_out,
            hidden,
            get_max_age(is_scaled=scaled_age),
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
            coral_output=True,
        )

    def forward(self, x):
        feat = self.e(x)
        out = self.c(feat)
        return {'age_pred': out}

    def init_params(self):
        print('running custom param init')
        self.e.init_params()
        self.c.apply(init_params())

    def end_of_epoch(self, epoch):
        self.e.end_of_epoch(epoch)


class ZAgeClassifier(Unit):
    def __init__(self,
        z_dims=512,
        hidden=[512, 256, 128],
        norm='batch',
        weight_norm=True,
        actv='mish',
        scaled_age=False,
    ):
        super().__init__()

        self.model = Mlp(
            z_dims,
            hidden,
            get_max_age(is_scaled=scaled_age),
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
            coral_output=True,
        )

    def forward(self, z):
        pred = self.model(z)
        return {
            'z_age_pred': pred,
        }
