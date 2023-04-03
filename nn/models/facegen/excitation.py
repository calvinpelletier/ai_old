#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import ConvToImg, ConvBlock
from ai_old.nn.blocks.mod import UpDoubleExcitationBlock, DoubleExcitationBlock
from ai_old.nn.blocks.blend import BlendBlock
from ai_old.util.etc import log2_diff


class BlendExcitationModulatedGenerator(Unit):
    def __init__(self,
        output_imsize=128,
        init_imsize=4,
        blend_start_imsize=64,
        nc_base=32,
        nc_max=512,
        z_dims=512,
        k_blend=1,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        n_down_up = log2_diff(output_imsize, init_imsize)
        n_blend = log2_diff(output_imsize, blend_start_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert max(nc) <= nc_max
        assert nc[0] == nc_base

        # for onnx exporter
        self.output_imsize = output_imsize
        self.init_imsize = init_imsize
        self.z_dims = z_dims
        self.nc = nc

        # initial
        self.initial = DoubleExcitationBlock(
            nc[-1],
            nc[-1],
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # up blocks
        up_blocks = []
        for i in range(n_down_up):
            up_blocks.append(UpDoubleExcitationBlock(
                nc[i+1],
                nc[i],
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.up_blocks = nn.ModuleList(up_blocks[::-1])

        # blend and bg blocks
        bg_blocks = [ConvBlock(
            3, # rgb
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]
        blend_blocks = [BlendBlock(
            nc[0],
            k=k_blend,
        )]
        for i in range(n_blend):
            bg_blocks.append(ConvBlock(
                nc[i],
                nc[i+1],
                s=2,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            blend_blocks.append(BlendBlock(
                nc[i+1],
                k=k_blend,
            ))
        self.bg_blocks = nn.ModuleList(bg_blocks)
        self.blend_blocks = nn.ModuleList(blend_blocks)

        # to img
        self.conv_to_img = ConvToImg(nc[0])

    def forward(self, init, z, bg):
        # encode background
        bg_feats = []
        for bg_block in self.bg_blocks:
            bg = bg_block(bg)
            bg_feats.append(bg)

        # generate
        x = self.initial(init, z)
        for i, up_block in enumerate(self.up_blocks):
            bg_idx = len(self.up_blocks) - i
            if bg_idx < len(bg_feats):
                x = self.blend_blocks[bg_idx](x, bg_feats[bg_idx])

            x = up_block(x, z)

        # final
        x = self.blend_blocks[0](x, bg_feats[0])
        return self.conv_to_img(x)


class ExcitationModulatedGenerator(Unit):
    def __init__(self,
        output_imsize=128,
        init_imsize=4,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        z_dims=512,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        n_down_up = log2_diff(output_imsize, init_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert max(nc) <= nc_max
        assert nc[0] == nc_base

        # up blocks
        up_blocks = []
        for i in range(n_down_up):
            up_blocks.append(UpDoubleExcitationBlock(
                nc[i+1],
                nc[i],
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.g = nn.ModuleList(up_blocks[::-1])

        # to img
        self.conv_to_img = ConvToImg(nc[0])

    def forward(self, init, z):
        x = init
        for dec in self.g:
            x = dec(x, z)
        return self.conv_to_img(x)
