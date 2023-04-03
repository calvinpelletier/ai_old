#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models import Unit
import ai_old.nn.blocks.stylegan as sg
import math


class StyleDiscriminator(Unit):
    def __init__(self,
        nc_in=3, # rgb
        imsize=128,
        # at res: 4    8    16   32   64  128
        channels=[512, 512, 256, 128, 64, 32],
    ):
        super().__init__()
        blur_kernel = [1, 3, 3, 1]

        assert 2**(len(channels) + 1) == imsize
        channels = {2**(i+2): x for i, x in enumerate(channels)}
        print(channels)

        convs = [sg.ConvLayer(nc_in, channels[imsize], 1)]
        in_channel = channels[imsize]
        for i in range(int(math.log(imsize, 2)), 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(sg.ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel
        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = sg.ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            sg.EqualLinear(
                channels[4] * 4 * 4,
                channels[4],
                activation='fused_lrelu',
            ),
            sg.EqualLinear(channels[4], 1),
        )

    def forward(self, input, label):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group,
            -1,
            self.stddev_feat,
            channel // self.stddev_feat,
            height,
            width,
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(batch, -1)
        out = self.final_linear(out)

        # dict because some discriminators return extra info
        return {f'd_{label}': out}
