#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.uenet import AdalinUpUenetBlock, SimpleUpUenetBlock
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock, ResDownConvBlock
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.encode import FeatMapToLatentViaFc, FeatMapToLatent


'''
name is because the model looks like UE when you draw it out

encodes to zspace, then decodes back up (with unet residuals) to an annotated
image (and optionally segments annotated image), and during decoding, spins out
per-resolution encodings to the modspace
'''
# TODO: try blur
class Uenet(Unit):
    def __init__(self,
        # backbone
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,
        n_layers_per_res=[2, 4, 8, 4, 2],
        to_mod_layers=2,

        # general
        z_dims=512,
        norm='batch',
        weight_norm=False,
        actv='mish',

        # generator specific
        g_dynamic_init=False,

        # segmentation
        segmentation=False,
    ):
        super().__init__()
        self.g_dynamic_init = g_dynamic_init

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down_up
        assert max(nc) <= z_dims
        assert z_dims == nc[-1]
        assert nc[0] == nc_base
        if input_imsize == 128 and smallest_imsize == 4:
            assert n_down_up == 5
            if nc_base == 32 and z_dims == 512:
                assert nc == [32, 64, 128, 256, 512, 512]

        # initial deepen
        self.initial = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # down/up blocks
        down_blocks = []
        up_blocks = []
        self.final_down = None
        for i in range(n_down_up):
            down = FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            )
            if i == n_down_up - 1:
                self.final_down = down
            else:
                down_blocks.append(down)

            up_blocks.append(AdalinUpUenetBlock(
                nc[i],
                nc[i-1] if i > 0 else nc[i],
                z_dims,
                to_mod_layers=to_mod_layers,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])

        # to z
        self.to_z = FeatMapToLatentViaFc(
            smallest_imsize,
            smallest_imsize,
            z_dims,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # segmentation
        # TODO
        assert not segmentation


    def forward(self, x):
        # deepen
        x = self.initial(x)

        # encode
        residuals = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            residuals.append(x)
        x = self.final_down(x)

        # save 4x4 to initialize generator
        if self.g_dynamic_init:
            g_init = x

        # calc latent
        z = self.to_z(x)

        # decode to modspace
        zz = [z]
        for up_block, res in zip(self.up_blocks, residuals[::-1]):
            x, mod = up_block(x, z, res)
            zz.append(z + mod)

        if self.g_dynamic_init:
            return zz, g_init
        return zz


class LightUenet(Unit):
    def __init__(self,
        # backbone
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,

        # general
        z_dims=512,
        norm='batch',
        weight_norm=False,
        actv='mish',

        # generator specific
        g_dynamic_init=False,

        # segmentation
        segmentation=False,
    ):
        super().__init__()
        self.g_dynamic_init = g_dynamic_init

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert max(nc) <= z_dims
        assert z_dims == nc[-1]
        assert nc[0] == nc_base
        if input_imsize == 128 and smallest_imsize == 4:
            assert n_down_up == 5
            if nc_base == 32 and z_dims == 512:
                assert nc == [32, 64, 128, 256, 512, 512]

        # initial deepen
        self.initial = ConvBlock(
            nc_in,
            nc[0],
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # down/up blocks
        down_blocks = []
        up_blocks = []
        self.final_down = None
        for i in range(n_down_up):
            down = ResDownConvBlock(
                nc[i],
                nc[i+1],
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            )
            if i == n_down_up - 1:
                self.final_down = down
            else:
                down_blocks.append(down)

            up_blocks.append(SimpleUpUenetBlock(
                nc[i],
                nc[i-1] if i > 0 else nc[i],
                z_dims,
                norm=norm,
                weight_norm=weight_norm,
                actv=actv,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])

        # to z
        self.to_z = FeatMapToLatent(
            smallest_imsize,
            smallest_imsize,
            z_dims,
            z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        # segmentation
        # TODO
        assert not segmentation


    def forward(self, x):
        # deepen
        x = self.initial(x)

        # encode
        residuals = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            residuals.append(x)
        x = self.final_down(x)

        # save 4x4 to initialize generator
        if self.g_dynamic_init:
            g_init = x

        # calc latent
        z = self.to_z(x)

        # decode to modspace
        zz = [z]
        for up_block, res in zip(self.up_blocks, residuals[::-1]):
            x, mod = up_block(x, z, res)
            zz.append(z + mod)

        if self.g_dynamic_init:
            return zz, g_init
        return zz

    def print_info(self):
        for down in self.down_blocks:
            n = 0
            for p in down.parameters():
                n += p.numel()
            print(f'down {n}')
        n = 0
        for p in self.final_down.parameters():
            n += p.numel()
        print(f'final down {n}')
        n = 0
        for p in self.to_z.parameters():
            n += p.numel()
        print(f'to z {n}')
        for up in self.up_blocks:
            n = 0
            for p in up.parameters():
                n += p.numel()
            print(f'up {n}')

        n_params = 0
        for p in self.parameters():
            n_params += p.numel()
        print('[INFO] built {} (total params: {})'.format(
            type(self).__name__,
            n_params,
        ))
        print(self)
