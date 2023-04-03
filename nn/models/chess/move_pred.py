#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.se import SEModule
from ai_old.util.params import init_params
from ai_old.nn.blocks.arcface import bottleneck_IR_SE


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


class ResnetBasedMovePredictor(nn.Module):
    def __init__(self,
        cfg,
        ctx_len=12,
        position_dim=12,
        meta_dim=7,
        move_dim=73,
        nc=64,
        n_blocks=6,
        norm='batch',
        actv='relu',
        block_type='maia',
        norm_first_layer=True,
    ):
        super().__init__()
        nc_in = meta_dim + position_dim * ctx_len
        self.flattened_move_dim = move_dim * 8 * 8

        # initial block
        blocks = [ConvBlock(
            nc_in,
            nc,
            norm=norm if norm_first_layer else 'none',
            actv=actv,
        )]

        # inner res blocks
        for i in range(n_blocks):
            if block_type == 'maia':
                blocks.append(_Block(nc, norm=norm, actv=actv))
            elif block_type == 'ir_se':
                blocks.apppend(bottleneck_IR_SE(nc, nc, 1))
            else:
                raise Exception(block_type)

        # final blocks
        blocks.append(ConvBlock(
            nc,
            nc,
            norm=norm,
            actv=actv,
        ))
        blocks.append(ConvBlock(
            nc,
            move_dim,
            norm='none',
            actv='none',
        ))

        self.net = nn.Sequential(*blocks)
        self.apply(init_params())

    def forward(self, x):
        move_pred = self.net(x)
        move_pred = torch.reshape(
            move_pred,
            (x.shape[0], self.flattened_move_dim),
        )
        return move_pred
