#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.util.params import init_params
from ai_old.nn.models.decode.style import StyleDecoder
from ai_old.nn.blocks.encode import FeatMapToLatentViaFc
from ai_old.nn.models.encode.fm2l import LearnedHybridFeatMapToLatent


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
        use_lhfmtl=False,
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
        if use_lhfmtl:
            self.to_z = LearnedHybridFeatMapToLatent(z_dims)
        else:
            self.to_z = FeatMapToLatentViaFc(
                smallest_imsize,
                smallest_imsize,
                z_dims,
                z_dims,
                norm=norm,
                weight_norm=False,
                actv=actv,
            )


    def forward(self, input):
        enc = self.net(input)
        z = self.to_z(enc)
        return enc, z


@persistence.persistent_class
class Autoencoder(nn.Module):
    def __init__(self,
        cfg,
        z_dims=512,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        norm='batch',
        full_bottleneck=False,
        e_actv='mish',
        e_layers_per_res=[2, 4, 8, 4, 2],
        e_use_lhfmtl=False,
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.intermediate = 'enc_plus_z'
        self.full_bottleneck = full_bottleneck

        self.e = Encoder(
            input_imsize=self.imsize,
            smallest_imsize=4,
            z_dims=z_dims,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
            n_layers_per_res=e_layers_per_res,
            norm=norm,
            actv=e_actv,
            use_lhfmtl=e_use_lhfmtl,
        )
        self.e.apply(init_params())

        self.g = StyleDecoder(
            imsize=self.imsize,
            smallest_imsize=smallest_imsize,
            z_dims=z_dims,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
            from_z_only=full_bottleneck,
        )
        self.g.apply(init_params())

    def forward(self, x):
        if self.full_bottleneck:
            _, z = self.e(x)
            return self.g(None, z)

        encoding, z = self.e(x)
        return self.g(encoding, z)

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.g.requires_grad_(True)
