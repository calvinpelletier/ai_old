#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OnnxConv2dLayer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        activation='linear',
        up=1,
        down=1,
        conv_clamp=None,
    ):
        super().__init__()

        if activation == 'swish':
            self.bias_act_fn = bias_act_swish
            self.act_gain = np.sqrt(2)
        elif activation == 'lrelu':
            self.bias_act_fn = bias_act_lrelu
            self.act_gain = np.sqrt(2)
        elif activation == 'linear':
            self.bias_act_fn = bias_act_linear
            self.act_gain = 1.
        else:
            raise Exception('TODO')

        # self.up = up
        # self.down = down
        assert up == 1, 'TODO'
        assert down == 1, 'TODO'

        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        weight = torch.randn([
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ])
        bias = torch.zeros([out_channels]) if bias else None
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x, gain=1.):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        x = conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            up=1,
            down=1,
            padding=self.padding,
            flip_weight=True,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain \
            if self.conv_clamp is not None else None
        x = self.bias_act_fn(x, b, gain=act_gain, clamp=act_clamp)
        return x


class OnnxEncSynthesisLayer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=3,
        activation='lrelu',
    ):
        super().__init__()
        self.padding = kernel_size // 2

        # self.act_gain = bias_act.activation_funcs[activation].def_gain
        if activation == 'swish':
            self.bias_act_fn = bias_act_swish
            self.act_gain = np.sqrt(2)
        elif activation == 'lrelu':
            self.bias_act_fn = bias_act_lrelu
            self.act_gain = np.sqrt(2)
        elif activation == 'linear':
            self.bias_act_fn = bias_act_linear
            self.act_gain = 1.
        else:
            raise Exception('TODO')

        self.affine = OnnxFullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = nn.Parameter(torch.randn([
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, gain=1):
        styles = self.affine(w)

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=None,
            up=1,
            padding=self.padding,
            flip_weight=True,
        )

        act_gain = self.act_gain * gain
        x = self.bias_act_fn(x, self.bias, gain=act_gain, clamp=None)
        return x


class OnnxFullyConnectedLayer(nn.Module):
    def __init__(self,
        in_features,
        out_features,
        bias_init=0,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn([out_features, in_features]),
        )
        self.bias = nn.Parameter(
            torch.full([out_features], np.float32(bias_init)))
        self.weight_gain = 1. / np.sqrt(in_features)

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        return torch.addmm(b.unsqueeze(0), x, w.t())


def modulated_conv2d(
    x,
    weight,
    styles,
    noise=None,
    up=1,
    down=1,
    padding=0,
    demodulate=True,
    flip_weight=True,
):
    batch_size = x.shape[0]

    # calculate per-sample weights and demodulation coefficient.
    w = None
    dcoefs = None
    if demodulate:
        w = weight.unsqueeze(0) # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1) # [NOIkk]
    if demodulate:
        # dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        dcoefs = ((w * w).sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]

    # execute by scaling the activations before and after the convolution
    x = x * styles.reshape(batch_size, -1, 1, 1)
    x = conv2d_resample(
        x=x,
        w=weight,
        up=up,
        down=down,
        padding=padding,
        flip_weight=flip_weight,
    )
    if demodulate and noise is not None:
        x = fma(
            x,
            dcoefs.reshape(batch_size, -1, 1, 1),
            noise,
        )
    elif demodulate:
        x = x * dcoefs.reshape(batch_size, -1, 1, 1)
    elif noise is not None:
        x = x.add_(noise)
    return x


def fma(a, b, c):
    return a * b + c


def conv2d_resample(
    x,
    w,
    up=1,
    down=1,
    padding=0,
    groups=1,
    flip_weight=True,
):
    if up > 1:
        w = w.transpose(0, 1)
        return _conv2d_wrapper(
            x=x,
            w=w,
            stride=up,
            padding=padding,
            groups=groups,
            transpose=True,
            flip_weight=(not flip_weight),
        )
    return _conv2d_wrapper(
        x=x,
        w=w,
        padding=padding,
        groups=groups,
        flip_weight=flip_weight,
    )


def _conv2d_wrapper(
    x,
    w,
    stride=1,
    padding=0,
    groups=1,
    transpose=False,
    flip_weight=True,
):
    if not flip_weight:
        w = w.flip([2, 3])

    op = _conv_transpose2d if transpose else _conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)


def _conv2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
):
    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def _conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    # output_padding=0,
    groups=1,
    dilation=1,
):
    return F.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        # output_padding=output_padding,
        output_padding=padding,
        groups=groups,
        dilation=dilation,
    )


def bias_act_swish(x, b, gain, clamp):
    if b is not None:
        x = x + b.reshape([1, -1, 1, 1])
    x = torch.sigmoid(x) * x
    x = x * gain
    if clamp is not None:
        x = x.clamp(-clamp, clamp)
    return x


def bias_act_lrelu(x, b, gain, clamp):
    if b is not None:
        x = x + b.reshape([1, -1, 1, 1])
    x = F.leaky_relu(x, 0.2)
    x = x * gain
    if clamp is not None:
        x = x.clamp(-clamp, clamp)
    return x


def bias_act_linear(x, b, gain, clamp):
    if b is not None:
        x = x + b.reshape([1, -1, 1, 1])
    if gain != 1.:
        x = x * gain
    if clamp is not None:
        x = x.clamp(-clamp, clamp)
    return x
