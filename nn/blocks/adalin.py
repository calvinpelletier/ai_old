#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.linear import EqualLinear
from ai_old.nn.blocks.norm import PixelNorm
from ai_old.nn.blocks.etc import get_actv
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.etc import Blur


class DownAdalinBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
        use_blur=False,
    ):
        super().__init__()

        # main flow
        self.main = NonResAdalinBlock(
            nc1,
            nc2,
            nc2,
            z_dims,
            s=2,
            actv=actv,
        )

        # residual-like shortcut
        shortcut_layers = [Blur()] if use_blur else []
        shortcut_layers.append(nn.AvgPool2d(2))
        shortcut_layers.append(ConvBlock(
            nc1,
            nc2,
            k=1,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        ))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x, z):
        return self.main(x, z) + self.shortcut(x)


class UpAdalinBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        k_shortcut=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        self.shortcut = ConvBlock(
            nc1,
            nc2,
            k=k_shortcut,
            norm=norm,
            weight_norm=weight_norm,
            actv='none',
        )

        self.main = NonResAdalinBlock(
            nc1,
            nc1,
            nc2,
            z_dims,
            actv=actv,
        )

        self.up = nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

    def forward(self, input, z):
        upsampled = self.up(input)
        return self.main(upsampled, z) + self.shortcut(upsampled)


class NonResAdalinBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        nc3,
        z_dims,
        s=1,
        k=3,
        actv='relu',
        pad='auto',
        use_bias=False,
    ):
        super().__init__()

        if pad == 'auto':
            pad = (k - 1) // 2

        self.pad1 = nn.ReflectionPad2d(pad)
        self.conv1 = nn.Conv2d(
            nc1, nc2,
            kernel_size=k,
            stride=s,
            padding=0,
            bias=use_bias,
        )
        self.norm1 = Adalin(nc2, z_dims)
        # self.actv = nn.ReLU(True)
        self.actv = get_actv(actv)

        self.pad2 = nn.ReflectionPad2d(pad)
        self.conv2 = nn.Conv2d(
            nc2, nc3,
            kernel_size=k,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        self.norm2 = Adalin(nc3, z_dims)

    def forward(self, x, z):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, z)
        out = self.actv(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, z)
        return out


class AdalinBlock(nn.Module):
    def __init__(self,
        nc,
        z_dims,
        k=3,
        actv='relu',
        pad='auto',
        use_bias=False,
    ):
        super().__init__()

        if pad == 'auto':
            pad = (k - 1) // 2

        self.pad1 = nn.ReflectionPad2d(pad)
        self.conv1 = nn.Conv2d(
            nc, nc,
            kernel_size=k,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        self.norm1 = Adalin(nc, z_dims)
        # self.actv = nn.ReLU(True)
        self.actv = get_actv(actv)

        self.pad2 = nn.ReflectionPad2d(pad)
        self.conv2 = nn.Conv2d(
            nc, nc,
            kernel_size=k,
            stride=1,
            padding=0,
            bias=use_bias,
        )
        self.norm2 = Adalin(nc, z_dims)

    def forward(self, x, z):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, z)
        out = self.actv(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, z)
        return out + x


class Adalin(nn.Module):
    def __init__(self, nc, z_dims, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, nc, 1, 1))
        self.rho.data.fill_(0.9)
        self.style_std = nn.Sequential(EqualLinear(z_dims, nc), PixelNorm())
        self.style_mean = nn.Sequential(EqualLinear(z_dims, nc), PixelNorm())


    def forward(self, input, style):
        bs = input.shape[0]

        gamma = 1 + self.style_std(style)
        beta = self.style_mean(style)

        in_mean = torch.mean(input, dim=[2, 3], keepdim=True)
        in_var = torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)

        ln_mean = torch.mean(input, dim=[1, 2, 3], keepdim=True)
        ln_var = torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + \
            (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + \
            beta.unsqueeze(2).unsqueeze(3)
        return out
