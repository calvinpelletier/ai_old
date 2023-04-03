#!/usr/bin/env python3
import torch
import torch.nn as nn
# from ai_old.nn.blocks.conv import ConvBlock
from external.sg2.unit import Conv2dLayer


class BlendBlock(nn.Module):
    def __init__(self,
        nc,
        k=1,
        use_fp16=False,
    ):
        super().__init__()
        self.use_fp16 = use_fp16

        self.to_mask = Conv2dLayer(
            nc,
            1,
            k,
            activation='sigmoid',
        )

    def forward(self, fg, bg):
        if self.use_fp16:
            bg = bg.to(torch.float16)
        mask = self.to_mask(fg)
        return fg * mask + bg * (1 - mask)


class MetaBlendBlock(nn.Module):
    def __init__(self,
        nc,
        k=1,
    ):
        super().__init__()
        self.to_mask = Conv2dLayer(
            nc,
            1,
            k,
            activation='sigmoid',
        )

    def forward(self, fg, bg, meta):
        mask = self.to_mask(meta)
        return fg * mask + bg * (1 - mask)


# class FancyBlendBlock(nn.Module):
#     def __init__(self, nc1):
#         super().__init__()
#
#         nc2 = min(nc1, 64)
#         nc3 = min(nc1, 32)
#         self.main = nn.Sequential(
#             ConvBlock(
#                 nc1,
#                 nc2,
#                 k=3,
#                 norm='batch',
#                 weight_norm=False,
#                 actv='mish',
#             ),
#             ConvBlock(
#                 nc2,
#                 nc3,
#                 k=3,
#                 norm='batch',
#                 weight_norm=False,
#                 actv='mish',
#             ),
#         )
#         self.shortcut = nn.Conv2d()
#
#         self.to_mask = ConvBlock(
#             nc3,
#             1,
#             k=k,
#             norm='none',
#             weight_norm=False,
#             actv='sigmoid',
#         )
#
#     def forward(self, fg, bg):
#         mask = self.to_mask(fg)
#         return fg * mask + bg * (1 - mask)
