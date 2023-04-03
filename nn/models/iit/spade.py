#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.models import Unit
from ai_old.nn.blocks.spade import UpSpadeResBlocks
from ai_old.nn.blocks.conv import ConvToImg, ConvBlock
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import ResDownConvBlock


class SpadeWithEncoderIIT(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=8,
        nc_in=3, # rgb
        nc_inner=256,
        nc_last=32, # nc into the ConvToImg
        k_init=5,
        k_down=3,
        norm='batch',
        weight_norm=False,
        use_spectral_norm_in_spade=True,
        actv='mish',
        use_blur=True,
    ):
        super().__init__()

        # encoder
        n_encode = log2_diff(imsize, smallest_imsize)
        nc_init = nc_inner // (2**n_encode)
        encoder_layers = [ConvBlock(
            3,
            nc_init,
            k=k_init,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]
        nc = nc_init
        for _ in range(n_encode):
            encoder_layers.append(ResDownConvBlock(
                nc,
                nc * 2,
                k_down=k_down,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                use_blur=use_blur,
            ))
            nc *= 2
        assert nc == nc_inner
        self.encoder = nn.Sequential(*encoder_layers)

        # generator
        self.generator = UpSpadeResBlocks(
            imsize=imsize,
            smallest_imsize=smallest_imsize,
            nc_init=nc_inner,
            nc_last=nc_last,
            norm=norm,
            use_spectral_norm=use_spectral_norm_in_spade,
        )

        # final layer
        self.conv_to_img = ConvToImg(nc_last)

    def forward(self, input):
        embedding = self.encoder(input)
        out = self.generator(embedding, input)
        return self.conv_to_img(out)


class SpadeIIT(Unit):
    def __init__(self,
        imsize=128,
        smallest_imsize=8,
        nc_in=3, # rgb
        nc_init=256,
        nc_last=32, # nc into the ConvToImg
        norm='batch',
        use_spectral_norm_in_spade=True,
    ):
        super().__init__()
        self.smallest_imsize = smallest_imsize

        self.preprocessor = nn.Conv2d(nc_in, nc_init, 3, padding=1)
        self.generator = UpSpadeResBlocks(
            imsize=imsize,
            smallest_imsize=smallest_imsize,
            nc_init=nc_init,
            nc_last=nc_last,
            norm=norm,
            use_spectral_norm=use_spectral_norm_in_spade,
        )
        self.conv_to_img = ConvToImg(nc_last)

    def forward(self, input):
        # preprocess
        downsampled_input = F.interpolate(
            input,
            size=(self.smallest_imsize, self.smallest_imsize),
            mode='bilinear',
            align_corners=True,
        )
        embedding = self.preprocessor(downsampled_input)

        # spade
        out = self.generator(embedding, input)

        # to img
        return self.conv_to_img(out)
