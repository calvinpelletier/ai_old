#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
# from kornia import filter2D


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_actv(actv):
    if actv == 'relu':
        return nn.ReLU()
    elif actv == 'mish':
        return Mish()
    elif actv == 'prelu':
        return nn.PReLU()
    elif actv == 'lrelu':
        return nn.LeakyReLU()
    elif actv == 'glu':
        return nn.GLU(dim=1)
    elif actv == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise Exception('unknown activation')


class CoralLayer(nn.Module):
    def __init__(self, dims_in, dims_out):
        super().__init__()
        self.coral_w = nn.Linear(dims_in, 1, bias=False)
        self.coral_b = nn.Parameter(torch.zeros(dims_out).float())

    def forward(self, x):
        return self.coral_w(x) + self.coral_b


# TODO
class Blur(nn.Module):
    def forward(self, x):
        raise Exception('TODO')
