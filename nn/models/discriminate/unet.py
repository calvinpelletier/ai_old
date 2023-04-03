#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.unet import DownUnetBlock, UpUnetBlock
from ai_old.nn.blocks.res import ResBlocks
from ai_old.nn.blocks.conv import ConvBlock


# TODO: try blur
class UnetDiscriminator(Unit):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=16,
        nc_max=512,
        n_res=1, # num res blocks at the smallest imsize
        k_shortcut=3, # kernel size used in the unet block shortcuts
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [nc_in] + [
            min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up)
        ]

        # sanity
        if input_imsize == 128 and smallest_imsize == 4:
            assert n_down_up == 5
            if nc_in == 3 and nc_base == 16:
                assert nc == [3, 16, 32, 64, 128, 256]
        assert max(nc) <= nc_max

        # down/up blocks
        down_blocks = []
        up_blocks = []
        for i in range(n_down_up):
            down_blocks.append(DownUnetBlock(
                nc[i],
                nc[i+1],
                k_shortcut=k_shortcut,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            up_blocks.append(UpUnetBlock(
                nc[i+1],
                nc[i],
                k_shortcut=k_shortcut,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])

        # res blocks
        self.res_blocks = ResBlocks(
            nc[-1],
            n_res,
            norm='batch',
            weight_norm=weight_norm,
            actv=actv,
        )

        # encoder logits
        # TODO: make final enc layers dynamic
        assert smallest_imsize == 4
        pre_final_enc_layer_nc = nc[-1] // 2
        pre_final_enc_layer_imsize = 4
        self.enc_out = nn.Sequential(
            ConvBlock(
                nc[-1],
                pre_final_enc_layer_nc,
                k=1,
                s=1,
                pad=0,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ),
            nn.Conv2d(
                pre_final_enc_layer_nc,
                1,
                kernel_size=pre_final_enc_layer_imsize,
                stride=1,
                padding=0,
            ),
        )

        # decoder logits
        self.dec_out = nn.Conv2d(nc_in, 1, kernel_size=1, padding=0)

    def forward(self, x, label):
        # encode
        residuals = []
        for down_block in self.down_blocks:
            x, unet_res = down_block(x)
            residuals.append(unet_res)
        x = self.res_blocks(x)
        enc_out = self.enc_out(x).squeeze()

        # decode
        for up_block, res in zip(self.up_blocks, residuals[::-1]):
            x = up_block(x, res)
        dec_out = self.dec_out(x)

        return {f'd_{label}_enc': enc_out, f'd_{label}_dec': dec_out}
