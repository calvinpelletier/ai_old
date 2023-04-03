#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.util.etc import log2_diff
from ai_old.nn.blocks.res import FancyMultiLayerDownBlock
from ai_old.nn.blocks.conv import ConvBlock
from ai_old.util.params import init_params
from external.sg2.unit import FullyConnectedLayer
from ai_old.nn.models.encode.fm2l import LearnedHybridFeatMapToLatent
from ai_old.nn.models.facegen.fgbg import FgSynthesisNetwork, BgSynthesisNetwork


@persistence.persistent_class
class FgBgEncoder(nn.Module):
    def __init__(self,
        input_imsize=256,
        smallest_imsize=4,
        z_dims_fg=512,
        z_dims_bg=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        n_layers_per_res=[2, 2, 4, 8, 4, 2],
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
        if fmtl_type == 'learned_hybrid':
            self.to_z_fg = LearnedHybridFeatMapToLatent(z_dims_fg)
            self.to_z_bg = LearnedHybridFeatMapToLatent(z_dims_bg)
        else:
            raise Exception(fmtl_type)

    def forward(self, input):
        enc = self.net(input)
        z_fg = self.to_z_fg(enc)
        z_bg = self.to_z_bg(enc)
        return z_fg, z_bg


@persistence.persistent_class
class FgBgDecoder(nn.Module):
    def __init__(self,
        imsize=256,
        z_dims_fg=512,
        z_dims_bg=512,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        additional_resolutions=[],
    ):
        super().__init__()

        self.g_fg = FgSynthesisNetwork(
            w_dim=z_dims_fg,
            img_resolution=imsize,
            img_channels=3,
            channel_base=64 * imsize,
            channel_max=z_dims_fg,
            num_fp16_res=4,
            conv_clamp=256,
            fp16_channels_last=False,
            architecture='resnet',
            additional_resolutions=additional_resolutions,
        )

        self.g_bg = BgSynthesisNetwork(
            w_dim=z_dims_bg,
            img_resolution=imsize,
            img_channels=3,
            channel_base=64 * imsize,
            channel_max=z_dims_bg,
            num_fp16_res=4,
            conv_clamp=256,
            fp16_channels_last=False,
            architecture='resnet',
        )

    def forward(self, z_fg, z_bg):
        ws_bg = z_bg.unsqueeze(dim=1).repeat(1, self.g_bg.num_ws, 1)
        x_bg = self.g_bg(ws_bg)

        ws_fg = z_fg.unsqueeze(dim=1).repeat(1, self.g_fg.num_ws, 1)
        img, seg = self.g_fg(x_bg, ws_fg)
        return img, seg

    def prep_for_train_phase(self):
        self.requires_grad_(True)



@persistence.persistent_class
class FgBgAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        z_dims_fg=512,
        z_dims_bg=512,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        norm='batch',
        e_pri_actv='mish',
        e_pri_layers_per_res=[2, 2, 4, 8, 4, 2],
        e_pri_fmtl_type='learned_hybrid',
        g_additional_layers=[],
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize

        self.e = FgBgEncoder(
            input_imsize=self.imsize,
            smallest_imsize=4,
            z_dims_fg=z_dims_fg,
            z_dims_bg=z_dims_bg,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
            n_layers_per_res=e_pri_layers_per_res,
            norm=norm,
            actv=e_pri_actv,
            fmtl_type=e_pri_fmtl_type,
        )
        self.e.apply(init_params())

        self.g = FgBgDecoder(
            imsize=self.imsize,
            z_dims_fg=z_dims_fg,
            z_dims_bg=z_dims_bg,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
            additional_resolutions=g_additional_layers,
        )
        self.g.apply(init_params())

    def forward(self, img):
        z_fg, z_bg = self.e(img)
        return self.g(z_fg, z_bg)

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.g.requires_grad_(True)
