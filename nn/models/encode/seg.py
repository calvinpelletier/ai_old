#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.nn.models import Unit
from ai_old.nn.blocks.conv import ConvToImg
from ai_old.nn.models.encode.simple import SimpleEncoder
from ai_old.nn.models.iit.unet import UnetIIT


class SegEncoder(Unit):
    def __init__(self,
        input_imsize=128,
        output_imsize=32,
        smallest_imsize=8,
        k=3,
        k_init=1,
        nc_in=19,
        nc_initial_base=32,
        nc_initial_out=32,
        nc_out=13,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.nc_in = nc_in

        modules = [SimpleEncoder(
            input_imsize=input_imsize,
            output_imsize=output_imsize,
            k=k,
            k_init=k_init,
            nc_in=nc_in,
            nc_base=nc_initial_base,
            nc_out=nc_initial_out,
            normalize_output=False,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )]

        if smallest_imsize is not None:
            modules.append(UnetIIT(
                input_imsize=output_imsize,
                smallest_imsize=smallest_imsize,
                nc_in=nc_initial_out,
                nc_out=nc_initial_out,
                nc_base=nc_initial_out,
                nc_max=512,
                n_res=1,
                k_shortcut=k,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
                normalize_output=False,
            ))

        modules.append(ConvToImg(nc_initial_out, nc_out=nc_out))

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        x = F.one_hot(x.to(torch.int64), num_classes=self.nc_in)
        b, h, w, c = x.shape
        x = torch.reshape(x, (b, c, h, w)).to(torch.float32)
        return self.model(x)


# to train the seg encoder
class SegEncoderWithIIT(Unit):
    def __init__(self,
        input_imsize=128,
        output_imsize=32,
        smallest_imsize=8,
        k=3,
        k_init=1,
        nc_in=19,
        nc_initial_base=32,
        nc_initial_out=32,
        nc_out=13,
        norm='batch',
        weight_norm=False,
        actv='mish',
    ):
        super().__init__()
        self.imsize = input_imsize

        self.seg_enc = SegEncoder(
            input_imsize=input_imsize,
            output_imsize=output_imsize,
            smallest_imsize=smallest_imsize,
            k=k,
            k_init=k_init,
            nc_in=nc_in,
            nc_initial_base=nc_initial_base,
            nc_initial_out=nc_initial_out,
            nc_out=nc_out,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        self.iit = UnetIIT(
            input_imsize=self.imsize,
            nc_in=16, # 3 (rgb) + 13 (seg enc)
        )

    def forward(self, x, seg):
        # encode
        enc = self.seg_enc(seg)

        # upsample and concat
        enc = F.interpolate(
            enc,
            size=(self.imsize, self.imsize),
            mode='bilinear',
            align_corners=True,
        )
        concat = torch.cat((x, enc), dim=1)

        # iit
        return self.iit(concat)
