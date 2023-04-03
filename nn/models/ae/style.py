#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.params import init_params
from ai_old.nn.models.facegen.low_res import LowResSynthesisNetwork
from ai_old.nn.models.encode.e4e import E4eEncoder
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.nn.blocks.res import ResDownConvBlock, FancyMultiLayerDownBlock
from ai_old.nn.blocks.encode import FeatMapToLatent, FeatMapToLatentViaFc


class LowResStyleGenerator(nn.Module):
    def __init__(self, imsize, nc_base=32, noise_mode='random'):
        super().__init__()
        self.noise_mode = noise_mode

        self.net = LowResSynthesisNetwork(
            imsize,
            channel_base=nc_base * 1024,
            num_fp16_res=4,
            conv_clamp=256,
        )

    def forward(self, ws, noise_mode_override=None):
        noise_mode = noise_mode_override if noise_mode_override is not None \
            else self.noise_mode
        out = self.net(ws, noise_mode=noise_mode)
        return out


class SimpleWEncoder(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        n_styles=18,
        z_dims=512,
        k=3,
        nc_in=3,
        nc_base=32,
        im2vec='mlp',
        norm='batch',
        actv='mish',
        norm_first_block=True,
        use_fancy_blocks=False,
        n_layers_per_res=[2, 4, 8, 4, 2],
    ):
        super().__init__()
        self.n_styles = n_styles

        n_down = log2_diff(input_imsize, smallest_imsize)
        nc = [min(z_dims, nc_base * 2**i) for i in range(n_down + 1)]

        if use_fancy_blocks:
            assert len(n_layers_per_res) == n_down

        # deepen
        layers = [ConvBlock(
            nc_in,
            nc[0],
            k=k,
            norm=norm if norm_first_block else 'none',
            actv=actv,
        )]

        # downsample
        for i in range(n_down):
            if use_fancy_blocks:
                layers.append(FancyMultiLayerDownBlock(
                    nc[i],
                    nc[i+1],
                    n_layers=n_layers_per_res[i],
                    norm=norm,
                    actv=actv,
                ))
            else:
                layers.append(ResDownConvBlock(
                    nc[i],
                    nc[i+1],
                    k_down=k,
                    norm=norm,
                    actv=actv,
                ))

        # final
        if im2vec == 'kernel':
            im2vec_module = FeatMapToLatent
        elif im2vec == 'mlp':
            im2vec_module = FeatMapToLatentViaFc
        else:
            raise ValueError(im2vec)

        layers.append(im2vec_module(
            smallest_imsize,
            4,
            nc[-1],
            z_dims,
            norm=norm,
            actv=actv,
        ))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).unsqueeze(1).repeat(1, self.n_styles, 1)


@persistence.persistent_class
class StyleAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        e_type='light_e4e',
        e_nc_base=32,
        e_norm_first_block=True,
        e_use_fancy_blocks=False,
        e_n_layers_per_res=[2, 4, 8, 4, 2],
        g_type='low_res_style',
        g_nc_base=32,
        g_noise_mode='random',
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.intermediate = 'modspace'

        if e_type == 'light_e4e':
            self.e = LightE4eEncoder(
                input_imsize=self.imsize,
                n_styles=18,
                noise_mode=g_noise_mode,
            )
        elif e_type == 'simple_w':
            self.e = SimpleWEncoder(
                input_imsize=self.imsize,
                n_styles=18,
                nc_base=e_nc_base,
                norm_first_block=e_norm_first_block,
                use_fancy_blocks=e_use_fancy_blocks,
                n_layers_per_res=e_n_layers_per_res,
            )
        self.e.apply(init_params())

        assert g_type == 'low_res_style'
        self.g = LowResStyleGenerator(
            self.imsize,
            nc_base=g_nc_base,
        )
        # self.g.apply(init_params())

    def forward(self, x):
        ws = self.e(x)
        return self.g(ws, noise_mode_override='const')

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.g.requires_grad_(True)
