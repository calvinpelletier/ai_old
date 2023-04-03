#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import normalize_2nd_moment, FullyConnectedLayer
import torch


class DynamicLerper(nn.Module):
    def __init__(self, input_dims, n_layers=8, lr_mul=1.):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(FullyConnectedLayer(
                input_dims if i == 0 else 512,
                512,
                activation='linear' if i == n_layers - 1 else 'lrelu',
                lr_multiplier=lr_mul,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LevelsDynamicLerper(nn.Module):
    def __init__(self,
        input_dims,
        n_layers=8,
        levels=['coarse', 'medium', 'fine'],
        mult=0.1,
        lr_mul=1.,
        gendered=True,
    ):
        super().__init__()
        self.gendered = gendered
        self.mult = mult
        self.coarse_enabled = 'coarse' in levels
        self.medium_enabled = 'medium' in levels
        self.fine_enabled = 'fine' in levels

        if self.coarse_enabled:
            self.coarse_lerper = DynamicLerper(
                input_dims,
                n_layers=n_layers,
                lr_mul=lr_mul,
            )
        if self.medium_enabled:
            self.medium_lerper = DynamicLerper(
                input_dims,
                n_layers=n_layers,
                lr_mul=lr_mul,
            )
        if self.fine_enabled:
            self.fine_lerper = DynamicLerper(
                input_dims,
                n_layers=n_layers,
                lr_mul=lr_mul,
            )

    def forward(self, w, gender, mag=1.):
        deltas = []
        for i in range(18):
            if i < 4:
                if self.coarse_enabled:
                    deltas.append(self.coarse_lerper(w[:, i, :]).unsqueeze(1))
            elif i < 8:
                if self.medium_enabled:
                    deltas.append(self.medium_lerper(w[:, i, :]).unsqueeze(1))
            else:
                if self.fine_enabled:
                    deltas.append(self.fine_lerper(w[:, i, :]).unsqueeze(1))
        delta = torch.cat(deltas, dim=1)
        delta = delta * self.mult * mag
        if self.gendered:
            gender_mult = (gender * 2. - 1.).unsqueeze(dim=1)
            delta = delta * gender_mult
        return delta


@persistence.persistent_class
class LerpGen(nn.Module):
    def __init__(self, cfg, z_dims=512, n_layers=8, lr_mul=1.):
        super().__init__()
        self.z_dims = z_dims

        self.initial = FullyConnectedLayer(512, 512, lr_multiplier=lr_mul)

        self.net = LevelsDynamicLerper(
            512 + z_dims,
            n_layers=n_layers,
            lr_mul=lr_mul,
        )

    def forward(self, z, w, gender, mag=1.):
        z_norm = normalize_2nd_moment(z).unsqueeze(1).repeat(1, 18, 1)
        w_norm = torch.zeros_like(w)
        for i in range(18):
            w_norm[:, i, :] = normalize_2nd_moment(self.initial(w[:, i, :]))
        x = torch.cat([w_norm, z_norm], dim=2)
        delta = self.net(x, gender, mag=mag)
        return w + delta

    def prep_for_train_phase(self):
        self.requires_grad_(True)


class SimplestLerpGen(nn.Module):
    def __init__(self, z_dims=512, n_layers=8, lr_mul=1.):
        super().__init__()
        self.initial = nn.Linear(z_dims, 512)
        self.net = DynamicLerper(512, n_layers=n_layers, lr_mul=lr_mul)

    def forward(self, z, w):
        delta = self.net(self.initial(z))
        return w + delta.unsqueeze(1).repeat(1, 18, 1)
