#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2.unit import FullyConnectedLayer
from ai_old.util.params import init_params


class HairDeltaGenerator(nn.Module):
    def __init__(self, z_dims=8):
        super().__init__()
        self.n_latents = 18
        w_dims = 512

        self.initial = nn.Sequential(
            FullyConnectedLayer(z_dims, w_dims, activation='lrelu'),
            FullyConnectedLayer(w_dims, w_dims, activation='lrelu'),
            FullyConnectedLayer(w_dims, w_dims, activation='lrelu'),
            FullyConnectedLayer(w_dims, w_dims, activation='lrelu'),
        )

        per_latent = []
        for i in range(self.n_latents):
            per_latent.append(
                FullyConnectedLayer(w_dims, w_dims, activation='linear'))
        self.per_latent = nn.ModuleList(per_latent)

        self.apply(init_params())

    def forward(self, w, z):
        x = self.initial(z)
        deltas = []
        for i in range(self.n_latents):
            deltas.append(self.per_latent[i](x).unsqueeze(dim=1))
        delta = torch.cat(deltas, dim=1)
        new_w = w + delta
        return new_w, delta
