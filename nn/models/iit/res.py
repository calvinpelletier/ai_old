#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import ConvBlock, DownConvBlocks, UpConvBlocks, ConvToImg
from ai_old.nn.blocks.res import ResBlocks
from ai_old.util.etc import log2_diff


# residual network for image-to-image translation
class ResIIT(Unit):
    def __init__(self,
        imsize=128, # input image size
        nc_init=64, # num channels after first layer
        k_init=7, # kernel size in first layer
        n_res=6, # num res blocks
        inner_imsize=32, # imsize during res blocks
        actv='mish', # activation function
        norm='batch', # normalization layer
        weight_norm=False, # normalize weights
    ):
        super().__init__()

        # initial layer to deepen the image
        initial_layer = ConvBlock(
            3, # initial channels (RGB)
            nc_init, # output channels
            k=k_init, # kernel size
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # downsample and deepen the image
        assert inner_imsize < imsize
        n_down = log2_diff(imsize, inner_imsize)
        down_layers = DownConvBlocks(
            nc_init, # initial channels
            n_down, # num downsampling layers
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )
        nc = nc_init * 2**n_down # num output channels

        # residual blocks
        res_layers = ResBlocks(
            nc,
            n_res,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # upsample and undeepen the image
        up_layers = UpConvBlocks(
            nc,
            n_down, # n_down == n_up
            norm=norm,
            actv=actv,
        )

        # convert to RGB
        output_layer = ConvToImg(nc_init)

        self.model = nn.Sequential(
            initial_layer,
            down_layers,
            res_layers,
            up_layers,
            output_layer,
        )

    def forward(self, x):
        return self.model(x)
