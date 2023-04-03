#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ai_old.nn.blocks.etc import get_actv
from ai_old.nn.blocks.norm import get_norm_1d


class LinearBlock(nn.Module):
    def __init__(self,
        dims_in,
        dims_out,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        norm, needs_bias = get_norm_1d(norm)
        linear = EqualLinear if weight_norm else nn.Linear

        components = [linear(dims_in, dims_out, bias=needs_bias)]
        if norm is not None:
            components.append(norm(dims_out))
        if actv != 'none':
            components.append(get_actv(actv))
        self.net = nn.Sequential(*components)

    def forward(self, x):
        return self.net(x)


class EqualLinear(nn.Module):
    def __init__(self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        lr_mul=1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.bias is not None:
            out = F.linear(
                input,
                self.weight * self.scale,
                bias=self.bias * self.lr_mul,
            )
        else:
            out = F.linear(input, self.weight * self.scale)

        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.weight.shape[1]}, \
            {self.weight.shape[0]})'
