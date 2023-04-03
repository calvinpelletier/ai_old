#!/usr/bin/env python3
from external.sg2 import persistence
from enum import Enum
import math
import numpy as np
import torch
import torch.nn as nn
from external.e4e.models.encoders.helpers import get_blocks, bottleneck_IR, \
    bottleneck_IR_SE, _upsample_add
from external.sg2.unit import FullyConnectedLayer


@persistence.persistent_class
class GradualStyleBlock(nn.Module):
    def __init__(self, nc_in, nc_out, res):
        super().__init__()
        self.nc_out = nc_out
        self.res = res
        num_pools = int(np.log2(res))
        modules = []
        modules += [
            nn.Conv2d(nc_in, nc_out, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(nc_out, nc_out, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = FullyConnectedLayer(nc_out, nc_out)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.nc_out)
        x = self.linear(x)
        return x


@persistence.persistent_class
class E4eEncoderNoProg(nn.Module):
    def __init__(self, imsize, num_layers=50, mode='ir_se'):
        super().__init__()
        assert num_layers in [50, 100, 152]
        assert mode in ['ir', 'ir_se']
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride,
                ))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(imsize, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        w0 = self.styles[0](c3)
        ws = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        features = c3
        for i in range(1, self.style_count):
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))
                features = p1
            delta = self.styles[i](features)
            ws[:, i] += delta
        return ws


@persistence.persistent_class
class E4eEncoderW0Only(nn.Module):
    def __init__(self, imsize, num_layers=50, mode='ir_se'):
        super().__init__()
        assert num_layers in [50, 100, 152]
        assert mode in ['ir', 'ir_se']
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride,
                ))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(imsize, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        w0 = self.styles[0](c3)
        ws = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        return ws


@persistence.persistent_class
class E4eEncoder(nn.Module):
    def __init__(self, input_imsize, n_styles):
        super().__init__()
        assert input_imsize == 256, 'TODO'

        num_layers = 50
        mode = 'ir_se'
        blocks = get_blocks(num_layers)
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride,
                ))
        self.body = nn.Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = n_styles
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = self.style_count

    def get_deltas_starting_dimensions(self):
        return list(range(self.style_count))

    def set_progressive_stage(self, new_stage):
        self.progressive_stage = new_stage
        print(f'changed progressive stage to: {new_stage}')

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w
