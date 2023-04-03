#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.se import SEModule
from ai_old.util.params import init_params
from ai_old.nn.blocks.arcface import bottleneck_IR_SE
from ai_old.nn.blocks.etc import Flatten


class _Block(nn.Module):
    def __init__(self, nc=64, norm='batch', actv='relu'):
        super().__init__()

        self.conv1 = ConvBlock(nc, nc, norm=norm, actv=actv)
        self.conv2 = ConvBlock(nc, nc, norm=norm, actv='none')
        self.se = SEModule(nc, 16)
        self.final = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        return self.final(x + res)


class ScorePredictor(nn.Module):
    def __init__(self,
        cfg,
        w=8,
        h=8,
        nc_in=None,
        nc=64,
        nc_last=8,
        n_blocks=6,
        norm='batch',
        actv='relu',
        block_type='maia',
        norm_first_layer=True,
        head_type='fc',
    ):
        super().__init__()

        blocks = [ConvBlock(
            nc_in,
            nc,
            norm=norm if norm_first_layer else 'none',
            actv=actv,
        )]
        for i in range(n_blocks):
            if block_type == 'maia':
                blocks.append(_Block(nc, norm=norm, actv=actv))
            elif block_type == 'ir_se':
                blocks.append(bottleneck_IR_SE(nc, nc, 1))
            else:
                raise Exception(block_type)
        blocks.append(ConvBlock(
            nc,
            nc_last,
            norm='none',
            actv='none',
        ))
        self.body = nn.Sequential(*blocks)

        if head_type == 'fc':
            self.head = nn.Sequential(
                Flatten(),
                nn.Linear(w * h * nc_last, 1),
            )
        else:
            raise Exception(head_type)

        self.apply(init_params())

    def forward(self, x):
        return self.head(self.body(x)).squeeze(1)
