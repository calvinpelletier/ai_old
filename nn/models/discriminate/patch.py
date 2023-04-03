#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import ConvBlock
import numpy as np


class MultiResPatchDiscriminator(Unit):
    def __init__(self,
        n_resolutions=2,
        nc_in=6, # defaults to dual rgb (for iit)
        nc_initial=64,
        nc_max=256,
        n_inner_layers=3,
        norm='batch',
        weight_norm=False,
        actv='lrelu',
    ):
        super().__init__()
        self.n = n_resolutions

        # one patch discriminator per resolution
        self.models = nn.ModuleList()
        for _ in range(self.n):
            self.models.append(_PatchDiscriminator(
                nc_in=nc_in,
                nc_initial=nc_initial,
                nc_max=nc_max,
                n_inner_layers=n_inner_layers,
                norm='batch',
                weight_norm=False,
                actv='lrelu',
            ))

    def forward(self, x, label):
        outputs = []
        features = []
        input = x
        for i in range(self.n):
            out, feat = self.models[i](input)
            outputs.append(out)
            features.append(feat)
            input = F.interpolate(
                input,
                scale_factor=0.5,
                mode='bilinear',
                align_corners=True,
                recompute_scale_factor=True,
            )
        return {f'd_{label}': outputs, f'd_{label}_feat': features}


# PatchGAN-style discriminator
class _PatchDiscriminator(Unit):
    def __init__(self,
        nc_in=6, # defaults to dual rgb (for iit)
        nc_initial=64,
        nc_max=256,
        n_inner_layers=3,
        norm='batch',
        weight_norm=False,
        actv='lrelu',
    ):
        super().__init__()
        nc_out = 1 # real/fake pred
        k = 4
        pad = int(np.ceil((k - 1.) / 2))
        nc = nc_initial
        self.net = nn.ModuleList()

        # initial layer
        self.net.append(ConvBlock(
            nc_in,
            nc,
            k=k,
            s=2,
            pad=pad,
            norm='none',
            weight_norm=False,
            actv=actv,
        ))

        # inner layers
        for i in range(n_inner_layers):
            prev = nc
            nc = min(nc * 2, nc_max)
            s = 2 if i < (n_inner_layers - 1) else 1
            self.net.append(ConvBlock(
                prev,
                nc,
                k=k,
                s=s,
                pad=pad,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            prev = nc

        # final layer
        self.net.append(nn.Conv2d(
            nc,
            nc_out,
            kernel_size=k,
            stride=1,
            padding=pad,
        ))

    def forward(self, input):
        outs = [input]
        for layer in self.net:
            outs.append(layer(outs[-1]))
        return outs[-1], outs[1:-1] # output, intermediate features
