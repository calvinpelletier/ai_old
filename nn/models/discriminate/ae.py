#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import ConvBlock, SimpleUpConvBlock, ConvToImg
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import ResDownConvBlock


# regularize the discriminator via an autoencoder, thus forcing it to extract
# meaningful image features
class AutoencoderDiscriminator(Unit):
    def __init__(self,
        input_imsize=128,
        inner_imsize=16, # resolution of image embedding for decode/discrim
        nc_in=3, # rgb
        nc_init=16,
        k_init=5,
        k_down=4,
        norm='batch',
        weight_norm=False,
        actv='mish',
        decoder_actv='glu',
        use_blur=False,
    ):
        super().__init__()
        n_encode_decode = log2_diff(input_imsize, inner_imsize)

        # encoder
        encoder_layers = [ConvBlock(
            3,
            nc_init,
            k=k_init,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]
        nc = nc_init
        for _ in range(n_encode_decode):
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
        self.encoder = nn.Sequential(*encoder_layers)
        nc_inner = nc # saving for discriminator

        # decoder
        decoder_layers = []
        for _ in range(n_encode_decode):
            decoder_layers.append(SimpleUpConvBlock(
                nc,
                nc // 2,
                norm=norm,
                weight_norm=weight_norm,
                actv=decoder_actv,
            ))
            nc //= 2
        decoder_layers.append(ConvToImg(nc))
        self.decoder = nn.Sequential(*decoder_layers)

        # discriminator
        pre_final_layer_imsize = 4
        n_discrim_down = log2_diff(inner_imsize, pre_final_layer_imsize)
        discrim_layers = []
        nc = nc_inner
        for _ in range(n_discrim_down):
            discrim_layers.append(ResDownConvBlock(
                nc,
                min(nc * 2, 512),
                k_down=k_down,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
            nc = min(nc * 2, 512)
        pre_final_layer_nc = nc // 2
        discrim_layers.append(ConvBlock(
            nc,
            pre_final_layer_nc,
            k=1,
            s=1,
            pad=0,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        ))
        discrim_layers.append(nn.Conv2d(
            pre_final_layer_nc,
            1,
            kernel_size=pre_final_layer_imsize,
            stride=1,
            padding=0,
        ))
        self.discrim = nn.Sequential(*discrim_layers)

    def forward(self, x, label):
        embedding = self.encoder(x)
        if label == 'real':
            rec = self.decoder(embedding)
        elif label == 'fake':
            rec = None
        else:
            raise Exception('wat')
        out = self.discrim(embedding)
        return {f'd_{label}': out, f'd_{label}_rec': rec}
