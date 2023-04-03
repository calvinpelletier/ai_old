#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.blocks.norm import get_norm
import math
from ai_old.nn.blocks.etc import get_actv
from external.sg2 import persistence
from external.sg2.unit import Conv2dLayer
from external.op import upfirdn2d
from external.op import bias_act
import numpy as np
from external.op import conv2d_resample
from ai_old.nn.blocks.onnx import OnnxConv2dLayer


class CustomConvBlock(nn.Module):
    def __init__(self,
        nc1, # input channels
        nc2, # output channels
        k=3, # kernel size
        s=1, # stride
        norm='batch',
        actv='swish',
        resample_filter=[1,3,3,1],
        pad='auto',
        bias='auto',
        conv_clamp=None,
        onnx=False,

        # deprecated
        weight_norm=False,
    ):
        super().__init__()

        # backwards compatibility
        if actv == 'none':
            actv = 'linear'
        elif actv == 'mish':
            actv = 'swish'

        # no longer supported
        assert not weight_norm
        assert actv != 'glu'

        # norm/bias
        norm, bias_suggestion = get_norm(norm)
        if bias == 'auto':
            bias = bias_suggestion

        if onnx:
            # conv that's convertible to onnx
            assert pad == 'auto'
            conv = OnnxConv2dLayer(
                nc1, nc2,
                kernel_size=k,
                bias=bias,
                activation=actv,
                up=1,
                down=s,
                conv_clamp=conv_clamp,
            )
        else:
            if pad == 'auto':
                _Conv = Conv2dLayer
            elif pad == 0:
                _Conv = UnpaddedFastConv
            else:
                raise Exception(pad)
            conv = _Conv(
                nc1, nc2,
                kernel_size=k,
                bias=bias,
                activation=actv,
                up=1,
                down=s,
                resample_filter=resample_filter,
                conv_clamp=conv_clamp,
                channels_last=False,
                trainable=True,
            )

        if norm is None:
            self.net = conv
        else:
            self.net = nn.Sequential(conv, norm(nc2))

    def forward(self, x):
        return self.net(x)


class UnpaddedFastConv(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        kernel_size,
        bias=True,
        activation='linear',
        up=1,
        down=1,
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        channels_last=False,
        trainable=True,
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )
        self.weight_gain = 1 / np.sqrt(nc1 * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else \
            torch.contiguous_format
        weight = torch.randn([
            nc2,
            nc1,
            kernel_size,
            kernel_size,
        ]).to(memory_format=memory_format)
        bias = torch.zeros([nc2]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=0,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain \
            if self.conv_clamp is not None else None
        x = bias_act.bias_act(
            x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class ConvBlock(nn.Module):
    def __init__(self,
        nc1, # input channels
        nc2, # output channels
        k=3, # kernel size
        s=1, # stride
        norm='batch',
        weight_norm=False,
        actv='mish',
        pad='auto',
        pad_type='reflect', # zeros, reflect, replicate, circular
        bias='auto',
    ):
        super().__init__()
        norm, bias_suggestion = get_norm(norm)
        if bias == 'auto':
            bias = bias_suggestion

        conv = EqualConv if weight_norm else nn.Conv2d

        if pad == 'auto':
            pad = (k - 1) // 2
        if actv == 'glu':
            nc2 *= 2 # because it gets div by 2 in actv fn

        components = [conv(
            nc1, nc2,
            kernel_size=k,
            stride=s,
            padding=pad,
            bias=bias,
        )]

        if norm is not None:
            components.append(norm(nc2))

        if actv != 'none':
            components.append(get_actv(actv))

        self.net = nn.Sequential(*components)

    def forward(self, x):
        return self.net(x)


class CustomSmartDoubleConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k=3,
        s=1,
        norm='batch',
        actv='swish',
        resample_filter=[1,3,3,1],
        bias='auto',
        conv_clamp=None,

        # deprecated
        pad='auto',
        weight_norm=False,
    ):
        super().__init__()
        # backwards compatibility
        if actv == 'none':
            actv = 'linear'
        elif actv == 'mish':
            actv = 'swish'

        # no longer supported
        assert not weight_norm
        assert actv != 'glu'
        assert pad == 'auto'
        assert bias == 'auto'

        norm, bias_suggestion = get_norm(norm)

        if bias == 'auto':
            bias = bias_suggestion

        # norm -> deepen conv -> actv -> stride conv -> norm
        components = []
        if norm is not None:
            components.append(norm(nc1))
        components.append(Conv2dLayer(
            nc1, nc2,
            kernel_size=k,
            bias=bias,
            activation=actv,
            up=1,
            down=1,
            conv_clamp=conv_clamp,
            channels_last=False,
            trainable=True,
        ))
        components.append(Conv2dLayer(
            nc2, nc2,
            kernel_size=k,
            bias=bias,
            activation=actv,
            up=1,
            down=s,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=False,
            trainable=True,
        ))
        if norm is not None:
            components.append(norm(nc2))
        self.net = nn.Sequential(*components)

    def forward(self, x):
        return self.net(x)


class SmartDoubleConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k=3,
        s=1,
        norm='batch',
        actv='swish',
        resample_filter=[1,3,3,1],
        bias='auto',

        # deprecated
        pad='auto',
        weight_norm=False,
    ):
        super().__init__()
        assert not weight_norm
        assert actv != 'glu'
        assert actv != 'none'

        norm, bias_suggestion = get_norm(norm)

        if bias == 'auto':
            bias = bias_suggestion

        if pad == 'auto':
            pad = (k - 1) // 2

        # norm -> deepen conv -> actv -> stride conv -> norm
        components = []
        if norm is not None:
            components.append(norm(nc1))
        components.append(nn.Conv2d(
            nc1, nc2,
            kernel_size=k,
            stride=1,
            padding=pad,
            bias=bias,
        ))
        components.append(get_actv(actv))
        components.append(nn.Conv2d(
            nc2, nc2,
            kernel_size=k,
            stride=s,
            padding=pad,
            bias=bias,
        ))
        components.append(get_actv(actv))
        if norm is not None:
            components.append(norm(nc2))
        self.net = nn.Sequential(*components)

    def forward(self, x):
        return self.net(x)


class DoubleConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        nc3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(
                nc1,
                nc2,
                k=3,
                s=1,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ),
            ConvBlock(
                nc2,
                nc3,
                k=3,
                s=1,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            )
        )

    def forward(self, x):
        return self.net(x)


class CustomDoubleConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        nc3,
        norm='batch',
        weight_norm=False,
        actv='mish',
        conv_clamp=None,
        onnx=False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            CustomConvBlock(
                nc1,
                nc2,
                k=3,
                s=1,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
                onnx=onnx,
            ),
            CustomConvBlock(
                nc2,
                nc3,
                k=3,
                s=1,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                conv_clamp=conv_clamp,
                onnx=onnx,
            )
        )

    def forward(self, x):
        return self.net(x)


class ConvToImg(nn.Module):
    def __init__(self, nc_in, nc_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                nc_in,
                nc_out,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DownConvBlocks(nn.Module):
    def __init__(self,
        nc_in,
        n_down,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        layers = []
        nc = nc_in
        for _ in range(n_down):
            layers.append(ConvBlock(
                nc,
                nc * 2,
                s=2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            nc *= 2
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SimpleUpConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k=3,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(
                nc1,
                nc2,
                k=k,
                s=1,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            )
        )

    def forward(self, x):
        return self.net(x)


class VectorUpConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        k=4,
        norm='batch',
        actv='glu',
    ):
        super().__init__()
        norm, conv_bias = get_norm(norm)
        if actv == 'glu':
            nc2 *= 2 # because it gets div by 2 in actv fn
        components = [nn.ConvTranspose2d(
            nc1,
            nc2,
            kernel_size=k,
            stride=1,
            padding=0,
            output_padding=0,
            bias=conv_bias,
        )]
        if norm is not None:
            components.append(norm(nc2))
        if actv != 'none':
            components.append(get_actv(actv))
        self.net = nn.Sequential(*components)

    def forward(self, x):
        return self.net(x)


# TODO: support weight norm
class UpConvBlocks(nn.Module):
    def __init__(self,
        nc_in,
        n_up,
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        layers = []
        nc = nc_in
        for _ in range(n_up):
            layers.append(UpConvBlock(
                nc,
                nc // 2,
                norm=norm,
                actv=actv,
            ))
            nc //= 2
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# TODO: support weight norm
class UpConvBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        norm='batch',
        actv='mish',
    ):
        super().__init__()
        k = 3
        pad = (k - 1) // 2
        norm, conv_bias = get_norm(norm)
        components = [nn.ConvTranspose2d(
            nc1, nc2,
            kernel_size=k,
            stride=2,
            padding=pad,
            output_padding=pad,
            bias=conv_bias,
        )]
        if norm is not None:
            components.append(norm(nc2))
        if actv != 'none':
            components.append(get_actv(actv))
        self.net = nn.Sequential(*components)

    def forward(self, x):
        return self.net(x)


class EqualConv(nn.Module):
    def __init__(self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.weight.shape[1]}, {self.weight.shape[0]}, '
            f'{self.weight.shape[2]}, '
            f'stride={self.stride}, padding={self.padding})'
        )
