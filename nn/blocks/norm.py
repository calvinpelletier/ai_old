#!/usr/bin/env python3
import torch
import torch.nn as nn
from math import sqrt


# returns (norm, should_enable_conv_bias)
def get_norm(norm):
    if norm == 'pixel':
        return PixelNorm, True
    elif norm == 'instance':
        return nn.InstanceNorm2d, True
    elif norm == 'batch':
        return nn.BatchNorm2d, False
    elif norm == 'none':
        return None, True
    else:
        raise Exception('unimplemented norm')


# returns (norm, should_enable_conv_bias)
def get_norm_1d(norm):
    if norm == 'batch':
        return nn.BatchNorm1d, False
    elif norm == 'none':
        return None, True
    else:
        raise Exception('unimplemented norm')


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


# class EqualLR:
#     def __init__(self, name):
#         self.name = name
#
#     def compute_weight(self, module):
#         weight = getattr(module, self.name + '_orig')
#         fan_in = weight.data.size(1) * weight.data[0][0].numel()
#
#         return weight * sqrt(2 / fan_in)
#
#     @staticmethod
#     def apply(module, name):
#         fn = EqualLR(name)
#
#         weight = getattr(module, name)
#         del module._parameters[name]
#         module.register_parameter(name + '_orig', nn.Parameter(weight.data))
#         module.register_forward_pre_hook(fn)
#
#         return fn
#
#     def __call__(self, module, input):
#         weight = self.compute_weight(module)
#         setattr(module, self.name, weight)
#
#
# def equal_lr(module, name='weight'):
#     EqualLR.apply(module, name)
#
#     return module
