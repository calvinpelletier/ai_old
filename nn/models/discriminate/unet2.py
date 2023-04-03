import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from math import log2
from ai_old.nn.models import Unit
from ai_old.nn.blocks.unet_disc import *


class UnetDiscriminator(Unit):
    def __init__(self,
        imsize=128,
        nc_in=3,
        nc_base=16,
        nc_max=512,
    ):
        super().__init__()
        num_layers = int(log2(imsize) - 3)

        blocks = []
        nc = [nc_in] + [(nc_base) * (2 ** i) for i in range(num_layers + 1)]

        set_nc_max = partial(min, nc_max)
        nc = list(map(set_nc_max, nc))
        nc[-1] = nc[-2]

        chan_in_out = list(zip(nc[:-1], nc[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)
            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)
        self.down_blocks = nn.ModuleList(down_blocks)
        last_chan = nc[-1]

        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(imsize // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x, label):
        b, *_ = x.shape

        residuals = []
        for down_block in self.down_blocks:
            x, unet_res = down_block(x)
            residuals.append(unet_res)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return {f'd_{label}_enc': enc_out.squeeze(), f'd_{label}_dec': dec_out}
