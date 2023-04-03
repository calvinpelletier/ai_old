#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2.unit import FullyConnectedLayer, modulated_conv2d, Conv2dLayer
from external.sg2 import persistence
from external.op import upfirdn2d
from external.op import bias_act
import numpy as np


@persistence.persistent_class
class StyleEncLayer(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        down=2,
        z_dims=512,
        kernel_size=3,
        activation='lrelu',
        resample_filter=[1,3,3,1],
        conv_clamp=None,
    ):
        super().__init__()
        self.down = down
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        self.affine = FullyConnectedLayer(z_dims, nc1, bias_init=1)
        memory_format = torch.contiguous_format
        self.weight = nn.Parameter(torch.randn([
            nc2,
            nc1,
            kernel_size,
            kernel_size,
        ]).to(memory_format=memory_format))
        self.bias = nn.Parameter(torch.zeros([nc2]))

    def forward(self, x, z, fused_modconv=True, gain=1):
        styles = self.affine(z)

        x = modulated_conv2d(
            x=x,
            weight=self.weight,
            styles=styles,
            noise=None,
            up=1,
            down=self.down,
            padding=self.padding,
            resample_filter=self.resample_filter,
            flip_weight=False,
            fused_modconv=fused_modconv,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain \
            if self.conv_clamp is not None else None
        x = bias_act.bias_act(
            x,
            self.bias.to(x.dtype),
            act=self.activation,
            gain=act_gain,
            clamp=act_clamp,
        )
        return x


@persistence.persistent_class
class StyleResDownBlock(nn.Module):
    def __init__(self,
        nc1,
        nc2,
        z_dims,
        resample_filter=[1,3,3,1],
        conv_clamp=None,
        use_fp16=False,
    ):
        super().__init__()
        self.nc1 = nc1
        self.z_dims = z_dims
        self.use_fp16 = use_fp16
        self.register_buffer(
            'resample_filter',
            upfirdn2d.setup_filter(resample_filter),
        )

        self.conv0 = StyleEncLayer(
            nc1,
            nc2,
            z_dims=z_dims,
            down=2,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )

        self.conv1 = StyleEncLayer(
            nc2,
            nc2,
            z_dims=z_dims,
            down=1,
            conv_clamp=conv_clamp,
        )

        self.skip = Conv2dLayer(
            nc1,
            nc2,
            kernel_size=1,
            bias=False,
            down=2,
            resample_filter=resample_filter,
        )

    def forward(self, x, z, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else \
            torch.float32
        x = x.to(dtype=dtype)
        y = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x, z, fused_modconv=False)
        x = self.conv1(x, z, fused_modconv=False, gain=np.sqrt(0.5))
        x = y.add_(x)
        return x
