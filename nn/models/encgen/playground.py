#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.blocks.res import ResUpConvBlock, ResDownConvBlock
from ai_old.nn.blocks.conv import ConvToImg, ConvBlock
from ai_old.nn.blocks.adalin import UpAdalinBlock, DownAdalinBlock
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.encode import FeatMapToLatentViaFc
from ai_old.nn.models.encode.simple import SimpleZEncoder
from ai_old.nn.blocks.mod import UpDoubleExcitationBlock, DownDoubleExcitationBlock, \
    DownDoubleStyleBlock


class SimpleAutoencoder(Unit):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        norm='batch',
        weight_norm=False,
        enc_actv='mish',
        dec_actv='mish',
    ):
        super().__init__()

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert max(nc) <= nc_max
        assert nc[0] == nc_base

        # initial deepen
        self.initial = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=enc_actv,
        )

        # down/up blocks
        down_blocks = []
        up_blocks = []
        for i in range(n_down_up):
            down_blocks.append(ResDownConvBlock(
                nc[i],
                nc[i+1],
                norm=norm,
                weight_norm=weight_norm,
                actv=enc_actv,
            ))
            up_blocks.append(ResUpConvBlock(
                nc[i+1],
                nc[i],
                norm=norm,
                weight_norm=weight_norm,
                actv=dec_actv,
            ))
        self.enc = nn.Sequential(*down_blocks)
        self.dec = nn.Sequential(*up_blocks[::-1])

        # to img
        self.conv_to_img = ConvToImg(nc[0])

    def forward(self, x):
        deepened = self.initial(x)
        encoded = self.enc(deepened)
        decoded = self.dec(encoded)
        return self.conv_to_img(decoded)


class AdalinAutoencoder(Unit):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        z_dims=512,
        up_type='adalin',
        norm='batch',
        weight_norm=False,
        enc_actv='mish',
        dec_actv='mish',
    ):
        super().__init__()

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        if up_type == 'adalin':
            up_cls = UpAdalinBlock
        elif up_type == 'excitation':
            up_cls = UpDoubleExcitationBlock
        else:
            raise ValueError(up_type)

        # sanity
        assert max(nc) <= nc_max
        assert nc[0] == nc_base

        # initial deepen
        self.initial = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=enc_actv,
        )

        # down/up blocks
        down_blocks = []
        up_blocks = []
        for i in range(n_down_up):
            down_blocks.append(ResDownConvBlock(
                nc[i],
                nc[i+1],
                norm=norm,
                weight_norm=weight_norm,
                actv=enc_actv,
            ))
            up_blocks.append(up_cls(
                nc[i+1],
                nc[i],
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=dec_actv,
            ))
        self.enc = nn.Sequential(*down_blocks)
        self.dec = nn.ModuleList(up_blocks[::-1])

        # to z
        self.to_z = FeatMapToLatentViaFc(
            smallest_imsize,
            4,
            nc[-1],
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=enc_actv,
        )

        # to img
        self.conv_to_img = ConvToImg(nc[0])

    def forward(self, x):
        x = self.initial(x)
        x = self.enc(x)
        z = self.to_z(x)
        for dec in self.dec:
            x = dec(x, z)
        return self.conv_to_img(x)


class LightFullyModulatedAutoencoder(Unit):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        z_dims=512,
        up_type='adalin',
        down_type='adalin',
        norm='batch',
        weight_norm=False,
        to_z_actv='mish',
        enc_actv='mish',
        dec_actv='mish',
        return_identity=False,
    ):
        super().__init__()
        self.return_identity = return_identity

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        if up_type == 'adalin':
            up_cls = UpAdalinBlock
        elif up_type == 'excitation':
            up_cls = UpDoubleExcitationBlock
        else:
            raise ValueError(up_type)

        if down_type == 'adalin':
            down_cls = DownAdalinBlock
        elif down_type == 'excitation':
            down_cls = DownDoubleExcitationBlock
        elif down_type == 'style':
            down_cls = DownDoubleStyleBlock
        else:
            raise ValueError(down_type)

        # sanity
        assert max(nc) <= nc_max
        assert nc[0] == nc_base

        # z encoder
        self.to_z = SimpleZEncoder(
            input_imsize=input_imsize,
            smallest_imsize=4,
            z_dims=z_dims,
            k=3,
            k_init=3,
            nc_in=nc_in,
            nc_base=nc_base,
            im2vec='mlp',
            norm=norm,
            weight_norm=weight_norm,
            actv=to_z_actv,
        )

        # initial deepen
        self.initial = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=enc_actv,
        )

        # down/up blocks
        down_blocks = []
        up_blocks = []
        for i in range(n_down_up):
            down_blocks.append(down_cls(
                nc[i],
                nc[i+1],
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=enc_actv,
            ))
            up_blocks.append(up_cls(
                nc[i+1],
                nc[i],
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=dec_actv,
            ))
        self.enc = nn.ModuleList(down_blocks)
        self.dec = nn.ModuleList(up_blocks[::-1])

        # to img
        self.conv_to_img = ConvToImg(nc[0])

    def forward(self, x, prefix=None):
        z = self.to_z(x)
        x = self.initial(x)
        for enc in self.enc:
            x = enc(x, z)
        identity = x
        for dec in self.dec:
            x = dec(x, z)
        out = self.conv_to_img(x)

        if self.return_identity:
            return {f'{prefix}_identity': identity, f'{prefix}_rec': out}
        return out


class ConstIdLightFullyModulatedAutoencoder(Unit):
    def __init__(self,
        smallest_imsize=4,
    ):
        self.model = LightFullyModulatedAutoencoder(
            smallest_imsize=smallest_imsize,
            return_identity=True,
        )

    def forward(self, male, female):
        m_out = self.model(male, 'm')
        f_out = self.model(female, 'f')
        return {**m_out, **f_out}
