#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.norm import PixelNorm
from torch.autograd import grad as Grad
from torch.autograd import Function
from math import sqrt
from ai_old.nn.blocks.linear import EqualLinear


class StyleBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        style_dim,
        pad_type='reflect',
        norm='pixel',
        activation='lrelu',
        downsample=False,
        upsample=False,
    ):
        super().__init__()
        assert activation == 'lrelu'
        activation = nn.LeakyReLU(0.2, False)
        self.act0 = activation
        self.act1 = activation
        self.act_gain = sqrt(2)

        if norm == 'pixel':
            self.norm0 = PixelNorm()
            self.norm1 = PixelNorm()
        else:
            raise Exception('unimplemented norm in modconv')

        self.conv0 = ModConv(
            nc1,
            nc2,
            style_dim,
            k=3,
            upsample=upsample,
        )
        self.conv1 = ModConv(
            nc2,
            nc2,
            style_dim,
            k=3,
            downsample=downsample,
        )

    def forward(self, x, latent):
        out = self.conv0(x, latent)
        out = self.act0(out) * self.act_gain
        out = self.norm0(out)
        out = self.conv1(out, latent)
        out = self.act1(out) * self.act_gain
        out = self.norm1(out)
        return out


class ModConv(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        style_dim,
        k=3,
        pad='auto',
        pad_type='reflect', # zeros, reflect, replicate, circular
        downsample=False,
        upsample=False,
    ):
        super().__init__()
        assert k == 3
        self.upsample = upsample
        self.downsample = downsample
        self.k = k
        self.nc1 = nc1
        self.nc2 = nc2

        if pad == 'auto':
            pad = (k - 1) // 2

        # assert pad_type == 'reflect'
        # self.padding = nn.ReflectionPad2d(pad)

        self.weight = nn.Parameter(torch.Tensor(nc2, nc1, k, k))
        self.bias = nn.Parameter(torch.Tensor(1, nc2, 1, 1))

        self.conv = F.conv2d
        self.style_std = nn.Sequential(EqualLinear(style_dim, nc1), PixelNorm())
        # self.blur = Blur(nc2)

        if self.upsample:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        if self.downsample:
            self.down = nn.AvgPool2d(2)

        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, x, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.nc2, self.nc1, self.k, self.k)

        # print('latent', latent, latent.shape)
        s = 1 + self.style_std(latent).view(-1, 1, self.nc1, 1, 1)
        weight = s * weight
        d = torch.rsqrt((weight ** 2).sum(4).sum(3).sum(2) + 1e-5).view(-1, self.nc2, 1, 1, 1)
        weight = (d * weight).view(-1, self.nc1, self.k, self.k)

        if self.upsample:
            x = self.up(x)
        # if self.downsample:
        #     x = self.blur(x)

        b, _, h, w = x.shape
        x = x.view(1, -1, h, w)
        # x = self.padding(x)
        out = self.conv(x, weight, padding=1, groups=b).view(b, self.nc2, h, w) + self.bias

        if self.downsample:
            out = self.down(out)
        # if self.upsample:
        #     out = self.blur(out)

        return out



# class EqualLinear(nn.Module):
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         linear = nn.Linear(dim_in, dim_out)
#         linear.weight.data.normal_()
#         linear.bias.data.zero_()
#         self.linear = equal_lr(linear)
#
#     def forward(self, input):
#         return self.linear(input)


class _BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class _BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = _BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None
_blur = _BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return _blur(input, self.weight, self.weight_flip)
