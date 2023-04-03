#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.util.params import init_params
from ai_old.nn.models.decode.style import LearnedConstStyleDecoder, ZOnlyStyleDecoder
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.models.encode.fm2l import LearnedHybridFeatMapToLatent
from external.e4e.models.encoders.helpers import get_blocks, bottleneck_IR_SE
from ai_old.nn.models.encode.effnet import effnetv2_m, effnetv2_l


@persistence.persistent_class
class FeatMapToLatent(nn.Module):
    def __init__(self,
        imsize,
        nc_in,
        z_dims,
        actv='mish',
    ):
        super().__init__()
        self.z_dims = z_dims

        convs = []
        n_down = log2_diff(imsize, 1)
        nc = nc_in
        for i in range(n_down):
            next_nc = min(z_dims, nc * 2)
            convs.append(ConvBlock(
                nc,
                next_nc,
                s=2,
                norm='none',
                actv=actv,
            ))
            nc = next_nc
        self.convs = nn.Sequential(*convs)

        self.linear = FullyConnectedLayer(
            z_dims,
            z_dims,
            activation='linear',
            lr_multiplier=1,
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.z_dims)
        x = self.linear(x)
        return x


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        n_layers_per_res=[2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
        fmtl_type='simple',
    ):
        super().__init__()
        n_down_up = log2_diff(input_imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # sanity
        assert len(n_layers_per_res) == n_down_up
        assert nc[0] == nc_base
        if input_imsize == 128 and smallest_imsize == 4:
            assert n_down_up == 5
            if nc_base == 64 and nc_max == 512:
                assert nc == [64, 128, 256, 512, 512, 512]

        blocks = [ConvBlock(
            nc_in,
            nc[0],
            norm='none',
            weight_norm=False,
            actv=actv,
        )]
        for i in range(n_down_up):
            down = FancyMultiLayerDownBlock(
                nc[i],
                nc[i+1],
                n_layers=n_layers_per_res[i],
                norm=norm,
                weight_norm=False,
                actv=actv,
            )
            blocks.append(down)
        self.net = nn.Sequential(*blocks)

        # to z
        if fmtl_type == 'simple':
            self.to_z = FeatMapToLatent(
                smallest_imsize,
                nc[-1],
                z_dims,
                actv=actv,
            )
        elif fmtl_type == 'learned_hybrid':
            self.to_z = LearnedHybridFeatMapToLatent(z_dims)
        else:
            raise Exception(fmtl_type)

    def forward(self, input):
        enc = self.net(input)
        z = self.to_z(enc)
        return z


@persistence.persistent_class
class ResnetEncoder(nn.Module):
    def __init__(self,
        imsize=256,
        z_dims=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        fmtl_type='learned_hybrid',
    ):
        super().__init__()
        assert imsize == 256 and z_dims == 512 and nc_in == 3 and \
            nc_base == 64 and nc_max == 512

        blocks = get_blocks(50)
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride,
                ))
        self.body = nn.Sequential(*modules)

        n_down_up = log2_diff(16, 4)

        blocks = []
        for i in range(n_down_up):
            down = FancyMultiLayerDownBlock(
                512,
                512,
                n_layers=2,
                norm='batch',
                weight_norm=False,
                actv='mish',
            )
            blocks.append(down)
        self.head = nn.Sequential(*blocks)

        # to z
        if fmtl_type == 'simple':
            self.to_z = FeatMapToLatent(
                smallest_imsize,
                nc[-1],
                z_dims,
                actv=actv,
            )
        elif fmtl_type == 'learned_hybrid':
            self.to_z = LearnedHybridFeatMapToLatent(z_dims)
        else:
            raise Exception(fmtl_type)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.body(x)
        x = self.head(x)
        z = self.to_z(x)
        return z



@persistence.persistent_class
class FullBottleneckAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        z_dims=512,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        norm='batch',
        e_type='basic',
        e_pri_actv='mish',
        e_pri_layers_per_res=[2, 4, 8, 4, 2],
        e_pri_fmtl_type='simple',
        g_type='lcsd',
        g_additional_4x4s=0,
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.intermediate = 'zspace'

        if e_type == 'basic':
            self.e = Encoder(
                input_imsize=self.imsize,
                smallest_imsize=4,
                z_dims=z_dims,
                nc_in=nc_in,
                nc_base=nc_base,
                nc_max=nc_max,
                n_layers_per_res=e_pri_layers_per_res,
                norm=norm,
                actv=e_pri_actv,
                fmtl_type=e_pri_fmtl_type,
            )
        elif e_type == 'resnet':
            self.e = ResnetEncoder(
                imsize=self.imsize,
                z_dims=z_dims,
                nc_in=nc_in,
                nc_base=nc_base,
                nc_max=nc_max,
                fmtl_type=e_pri_fmtl_type,
            )
        elif e_type == 'effnet-m':
            self.e = effnetv2_m(num_classes=z_dims)
        elif e_type == 'effnet-l':
            self.e = effnetv2_l(num_classes=z_dims)
        else:
            raise Exception(e_type)
        self.e.apply(init_params())

        if g_type == 'lcsd':
            self.g = LearnedConstStyleDecoder(
                imsize=self.imsize,
                smallest_imsize=smallest_imsize,
                z_dims=z_dims,
                nc_in=nc_in,
                nc_base=nc_base,
                nc_max=nc_max,
                additional_4x4s=g_additional_4x4s,
            )
        elif g_type == 'zosd':
            assert g_additional_4x4s == 0, 'todo'
            self.g = ZOnlyStyleDecoder(
                imsize=self.imsize,
                smallest_imsize=smallest_imsize,
                z_dims=z_dims,
                nc_in=nc_in,
                nc_base=nc_base,
                nc_max=nc_max,
            )
        else:
            raise Exception(g_type)
        self.g.apply(init_params())

    def forward(self, x):
        z = self.e(x)
        return self.g(z)

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.g.requires_grad_(True)
