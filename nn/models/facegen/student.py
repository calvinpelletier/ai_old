#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from external.sg2 import persistence
import external.sg2.misc as misc


@persistence.persistent_class
class Sg2Student(nn.Module):
    def __init__(self, cfg, nc_base=32, nc_max=512):
        super().__init__()
        student_imsize = cfg.dataset.imsize

        self.net = SynthesisNetwork(
            student_imsize,
            channel_base=nc_base * 1024,
            channel_max=nc_max,
        )

    def forward(self, ws):
        out = self.net(ws, noise_mode='const')
        return out

    def prep_for_train_phase(self):
        self.net.requires_grad_(True)


class SynthesisNetwork(nn.Module):
    def __init__(self,
        student_imsize,
        w_dim=512,
        teacher_imsize=1024,
        img_channels=3,
        channel_base=32768,
        channel_max=512,
    ):
        assert teacher_imsize >= 4 and \
            teacher_imsize & (teacher_imsize - 1) == 0
        super().__init__()
        self.student_imsize = student_imsize
        self.w_dim = w_dim
        self.teacher_imsize = teacher_imsize
        self.img_resolution_log2 = int(np.log2(teacher_imsize))
        self.img_channels = img_channels
        self.block_resolutions = [
            2 ** i for i in range(2, self.img_resolution_log2 + 1)
        ]
        channels_dict = {
            res: min(channel_base // res, channel_max) \
            for res in self.block_resolutions
        }

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            is_last = (res == self.teacher_imsize)
            block = SynthesisBlock(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=res if res <= student_imsize else student_imsize,
                img_channels=img_channels,
                is_last=is_last,
                is_up=res <= student_imsize,
            )
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(
                    1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img


class SynthesisBlock(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        img_channels,
        is_last,
        is_up,
        architecture='skip',
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.is_up = is_up
        self.architecture = architecture
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = nn.Parameter(
                torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                up=2 if is_up else 1,
            )
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
        )
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2 if is_up else 1,
            )

    def forward(self, x, img, ws, **layer_kwargs):
        w_iter = iter(ws.unbind(dim=1))

        # input
        if self.in_channels == 0:
            x = self.const
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])

        # main layers
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(w_iter), **layer_kwargs)
            x = self.conv1(x, next(w_iter), gain=np.sqrt(0.5), **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(x, next(w_iter), **layer_kwargs)
            x = self.conv1(x, next(w_iter), **layer_kwargs)

        # to rgb
        if img is not None:
            if self.is_up:
                img = F.interpolate(
                    img,
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True,
                )
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter))
            img = img.add_(y) if img is not None else y

        return x, img


class SynthesisLayer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        resolution,
        kernel_size=3,
        up=1,
        use_noise=True,
        activation='lrelu',
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.padding = kernel_size // 2

        assert activation == 'lrelu'
        # self.activation = activation
        self.act_gain = np.sqrt(2)

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = nn.Parameter(torch.randn([
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]))
        if use_noise:
            self.register_buffer(
                'noise_const',
                torch.randn([resolution, resolution]),
            )
            self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, w, noise_mode='random', gain=1):
        styles = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution],
                device=x.device,
            ) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=noise,
            up=self.up,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        x = bias_act_lrelu(
            x,
            self.bias,
            gain=act_gain,
        )
        return x


class ToRGBLayer(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        w_dim,
        kernel_size=1,
    ):
        super().__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        self.weight = nn.Parameter(torch.randn([
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
        ]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            demodulate=False,
        )
        x = bias_act_linear(
            x,
            self.bias,
        )
        return x


class FullyConnectedLayer(nn.Module):
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


def fma(a, b, c):
    return a * b + c


@misc.profiled_function
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


@misc.profiled_function
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


def bias_act_lrelu(x, b, gain):
    x = x + b.reshape([1, -1, 1, 1])
    x = F.leaky_relu(x, 0.2)
    return x * gain


def bias_act_linear(x, b):
    return x + b.reshape([1, -1, 1, 1])
