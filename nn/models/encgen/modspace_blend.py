#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from ai_old.nn.models.encode.squeeze import ZEncoder
from ai_old.nn.models.encode.style import StyleEncoderToModspace
from ai_old.nn.models.encgen.old_blend import IbgStyleEncoder, BlendSynthesisNetwork
from ai_old.util.factory import build_model_from_exp
from ai_old.util.params import init_params
from ai_old.util import config
import copy


@persistence.persistent_class
class PriModIbgEncoders(nn.Module):
    def __init__(self,
        imsize=128,
        z_dims=512,
        nc_base=32,
        nc_max=512,
    ):
        super().__init__()

        self.e_pri = ZEncoder(
            input_imsize=imsize,
            smallest_imsize=4,
            z_dims=z_dims,
            nc_in=3,
            nc_base=nc_base,
            n_layers_per_res=[2, 4, 8, 4, 2],
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        self.e_mod = StyleEncoderToModspace(input_imsize=imsize)

        self.e_ibg = IbgStyleEncoder(
            img_resolution=imsize,
            img_channels=3,
            channel_base=nc_base * imsize,
            channel_max=nc_max,
            w_dim=z_dims,
        )

    def forward(self, fg, ibg):
        z = self.e_pri(fg)
        ws = self.e_mod(fg, z)
        ibg_encs = self.e_ibg(ibg, z)
        return ws, ibg_encs


@persistence.persistent_class
class ModspaceBlendAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        g_exp='facegen/8/1',
        g_k_blend=3,
    ):
        super().__init__()

        # load the original generator and its config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G')

        # hparams
        self.z_dims = config.get_default(og_cfg.model.G, 'z_dims', 512)
        self.num_ws = og_G.f.num_ws
        self.imsize = config.get_default(og_cfg.model.G, 'imsize', 128)
        nc_in = config.get_default(og_cfg.model.G, 'nc_in', 3)
        nc_base = config.get_default(og_cfg.model.G, 'nc_base', 32)
        nc_max = config.get_default(og_cfg.model.G, 'nc_max', 512)
        num_fp16_res = config.get_default(og_cfg.model.G, 'num_fp16_res', 4)

        # create blend synth network
        self.g = BlendSynthesisNetwork(
            w_dim=self.z_dims,
            img_resolution=self.imsize,
            img_channels=nc_in,
            channel_base=nc_base * self.imsize,
            channel_max=nc_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=256,
            fp16_channels_last=False,
            k_blend=g_k_blend,
        )
        self.g.requires_grad_(False)

        # transfer params from the original generator
        for res in self.g.block_resolutions:
            setattr(
                self.g,
                f'b{res}',
                copy.deepcopy(getattr(og_G.g, f'b{res}')),
            )

        del og_G

        # build encoder
        self.e = PriModIbgEncoders(
            imsize=self.imsize,
            z_dims=self.z_dims,
            nc_base=nc_base,
            nc_max=nc_max,
        )
        self.e.apply(init_params())

    def forward(self, fg, ibg):
        ws, ibg_encs = self.e(fg, ibg)
        output = self.g(ibg_encs, ws)
        return output

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
