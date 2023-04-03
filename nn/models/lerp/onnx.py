#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.norm import PixelNorm
import numpy as np


def bias_act_lrelu(x, b):
    alpha = 0.2
    gain = np.sqrt(2)
    x = x + b.reshape([1, -1])
    x = F.leaky_relu(x, alpha)
    return x * gain


class FullyConnectedLayer(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        activation='linear',
        lr_multiplier=1,
        bias_init=0,
    ):
        super().__init__()
        assert activation in ['linear', 'lrelu']
        self.activation = activation

        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier,
        )
        self.bias = torch.nn.Parameter(
            torch.full([out_features], np.float32(bias_init)))
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if self.bias_gain != 1:
            b = b * self.bias_gain

        if self.activation == 'linear':
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act_lrelu(x, b)
        return x


class OnnxDynamicLerper(nn.Module):
    def __init__(self, final_activation, lr_mul, pixel_norm):
        super().__init__()
        layers = [PixelNorm()] if pixel_norm else []
        n_layers = 4
        for i in range(n_layers):
            actv = final_activation if i == n_layers - 1 else 'lrelu'
            # assert actv in ['linear', 'lrelu']
            # actv = actv == 'lrelu'
            layers.append(FullyConnectedLayer(
                512,
                512,
                activation=actv,
                lr_multiplier=lr_mul,
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OnnxLevelsDynamicLerper(nn.Module):
    def __init__(self, levels, final_activation, mult, lr_mul, pixel_norm):
        super().__init__()
        self.mult = mult
        assert 'coarse' in levels
        assert 'medium' in levels
        assert 'fine' in levels
        self.coarse_lerper = OnnxDynamicLerper(
            final_activation, lr_mul, pixel_norm)
        self.medium_lerper = OnnxDynamicLerper(
            final_activation, lr_mul, pixel_norm)
        self.fine_lerper = OnnxDynamicLerper(
            final_activation, lr_mul, pixel_norm)

    def forward(self, w, gender, magnitude=1.):
        # delta = torch.zeros_like(w)
        # for i in range(18):
        #     if i < 4:
        #         if self.coarse_enabled:
        #             delta[:, i, :] = self.coarse_lerper(w[:, i, :])
        #     elif i < 8:
        #         if self.medium_enabled:
        #             delta[:, i, :] = self.medium_lerper(w[:, i, :])
        #     else:
        #         if self.fine_enabled:
        #             delta[:, i, :] = self.fine_lerper(w[:, i, :])

        delta = []
        for i in range(18):
            if i < 4:
                delta.append(self.coarse_lerper(w[:, i, :]).unsqueeze(1))
            elif i < 8:
                delta.append(self.medium_lerper(w[:, i, :]).unsqueeze(1))
            else:
                delta.append(self.fine_lerper(w[:, i, :]).unsqueeze(1))
        delta = torch.cat(delta, dim=1)

        delta = delta * self.mult * magnitude
        gender_mult = (gender * 2. - 1.).unsqueeze(dim=1)
        delta = delta * gender_mult

        return w + delta
