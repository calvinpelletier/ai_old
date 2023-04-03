#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.sg2 import DiscriminatorBlock, Flatten
from math import log2


class StyleDiscriminator(Unit):
    def __init__(self,
        imsize=128,
        nc_in=3,
        nc_base=64,
        nc_max=512,
    ):
        super().__init__()
        num_layers = int(log2(imsize))
        nc = [nc_in] + [
            min(nc_max, nc_base * (2 ** i)) for i in range(num_layers)
        ]

        blocks = []
        for i in range(len(nc) - 1):
            block = DiscriminatorBlock(
                nc[i],
                nc[i+1],
                downsample=i != (len(nc) - 2),
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.Conv2d(nc[-1], nc[-1], 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(2 * 2 * nc[-1], 1)

    def forward(self, x, label):
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return {f'd_{label}': x.squeeze()}
