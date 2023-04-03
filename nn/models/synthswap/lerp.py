#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.mlp import Mlp
import os
import ai_old.constants as c
import numpy as np


# baseline (non-trainable)
class StaticGenderLerper(Unit):
    def __init__(self, lerp_path='mtf/0.gz', lerp_mult=1.25):
        path = os.path.join(c.LERP_PATH, lerp_path)
        self.lerp = torch.tensor(
            np.loadtxt(path)).float().unsqueeze(0).cuda()
        self.lerp *= lerp_mult

    def forward(self, z1, gender1):
        sign = gender1 * 2. - 1. # 1=male,0=female -> 1=male,-1=female
        self.lerp = self.lerp.expand(z1.shape[0], -1)
        sign = sign.unsqueeze(1).expand(-1, self.lerp.shape[1])
        z2 = z1 + self.lerp * sign
        return {'z2': z2}


class DynamicGenderLerper(Unit):
    def __init__(self,
        base_lerp_path='mtf/0.gz',
        base_lerp_mult=1.25,
        z_dims=512,
        hidden=[512, 512, 512, 512, 512],
        norm='batch',
        weight_norm=True,
        actv='mish',
    ):
        super().__init__()

        # base lerp
        path = os.path.join(c.LERP_PATH, base_lerp_path)
        self.base_lerp = torch.tensor(
            np.loadtxt(path)).float().unsqueeze(0).cuda()
        self.base_lerp *= base_lerp_mult

        # dynamic adjustment
        self.model = Mlp(
            z_dims,
            hidden,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, z1, gender1):
        sign = gender1 * 2. - 1. # 1=male,0=female -> 1=male,-1=female
        self.base_lerp = self.base_lerp.expand(z1.shape[0], -1)
        sign = sign.unsqueeze(1).expand(-1, self.base_lerp.shape[1])
        base_z2 = z1 + self.base_lerp * sign
        adjustment = self.model(base_z2)
        z2 = base_z2 + adjustment
        return {
            'base_z2': base_z2,
            'z2': z2,
            'adj_mag': torch.linalg.norm(adjustment, dim=1, ord=1),
        }


'''
In contrast to the dynamic lerper above, which learns a single adjustment
dynamically calculated based on the incoming Z to counter any unwanted
attribute drift, this has 2 adjustments (one for each drifting attribute:
glasses and mouth) which each have a static direction (learned during training
but universal to all input Z) and dynamic magnitudes. Ideally, this will
force the model to learn the directions in the latent space that correspond to
those attributes and figure out which input Zs require an adjustment in that
direction. However, this might get stuck in a bad local minimum where both
directions are utilized for the easiest attribute and the other attribute is
ignored (try countering this by initializing the directions).
'''
class DynamicMagStaticDirGenderLerper(Unit):
    def __init__(self,
        base_lerp_path='mtf/0.gz',
        base_lerp_mult=1.25,
        initialize_directions=True,
        freeze_directions=True,
        z_dims=512,
        hidden=[512, 256, 128, 32, 8],
        norm='batch',
        weight_norm=True,
        actv='mish',
        adj_age=False,
    ):
        super().__init__()
        self.adj_age = adj_age

        # base lerp
        path = os.path.join(c.LERP_PATH, base_lerp_path)
        self.base_lerp = torch.tensor(
            np.loadtxt(path)).float().unsqueeze(0).cuda()
        self.base_lerp *= base_lerp_mult

        # (optionally learned) static adjustment directions
        if initialize_directions:
            # smile
            path1 = os.path.join(c.LERP_PATH, 'smile/1.gz')
            dir1 = torch.tensor(np.loadtxt(path1)).float()

            # glasses
            path2 = os.path.join(c.LERP_PATH, 'glasses/0.gz')
            dir2 = torch.tensor(np.loadtxt(path2)).float()

            if adj_age:
                # age
                path3 = os.path.join(c.LERP_PATH, 'age/1.gz')
                dir3 = torch.tensor(np.loadtxt(path3)).float()
        else:
            dir1 = torch.randn(z_dims)
            dir2 = torch.randn(z_dims)
            if adj_age:
                dir3 = torch.randn(z_dims)

        if freeze_directions:
            self.dir1 = dir1.to('cuda')
            self.dir2 = dir2.to('cuda')
            if adj_age:
                self.dir3 = dir3.to('cuda')
        else:
            self.dir1 = nn.Parameter(dir1).to('cuda')
            self.dir2 = nn.Parameter(dir2).to('cuda')
            if adj_age:
                self.dir3 = nn.Parameter(dir3).to('cuda')

        # dynamic adjustment magnitudes
        self.mag1 = self._create_mag(z_dims, hidden, norm, weight_norm, actv)
        self.mag2 = self._create_mag(z_dims, hidden, norm, weight_norm, actv)
        if adj_age:
            self.mag3 = self._create_mag(z_dims, hidden, norm, weight_norm, actv)

    def _create_mag(self, z_dims, hidden, norm, weight_norm, actv):
        return Mlp(
            z_dims,
            hidden,
            1,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

    def forward(self, z1, gender1):
        # base gender lerp
        sign = gender1 * 2. - 1. # 1=male,0=female -> 1=male,-1=female
        base_lerp = self.base_lerp.expand(z1.shape[0], -1)
        sign = sign.unsqueeze(1).expand(-1, base_lerp.shape[1])
        base_z2 = z1 + base_lerp * sign

        # adjusted lerp
        adj = sign * self.dir1 * self.mag1(base_z2) + \
              sign * self.dir2 * self.mag2(base_z2)
        if self.adj_age:
            adj += sign * self.dir3 * self.mag3(base_z2)
        z2 = base_z2 + adj

        return {
            'base_z2': base_z2,
            'z2': z2,
            'adj_mag': torch.linalg.norm(adj, dim=1, ord=1),
        }
