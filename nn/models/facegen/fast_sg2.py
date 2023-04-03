#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisNetwork, MappingNetwork


@persistence.persistent_class
class FastSg2Generator(nn.Module):
    def __init__(self,
        cfg,
        # NOTE: if you change these defaults, also change the tmp hack in
        # unit/encgen/blend.py:GeneratorInitializedBlendAutoencoder
        z_dims=512,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        num_fp16_res=4,
        f_n_layers=2,
    ):
        super().__init__()
        self.imsize = cfg.dataset.imsize
        self.z_dims = z_dims
        self.g = SynthesisNetwork(
            w_dim=z_dims,
            img_resolution=self.imsize,
            img_channels=nc_in,
            channel_base=nc_base * self.imsize,
            channel_max=nc_max,
            num_fp16_res=num_fp16_res,
            conv_clamp=256,
            fp16_channels_last=False,
        )
        self.num_ws = self.g.num_ws
        self.f = MappingNetwork(
            z_dim=z_dims,
            c_dim=0,
            w_dim=z_dims,
            num_ws=self.num_ws,
            num_layers=f_n_layers,
        )

    def forward(self,
        z,
        truncation_psi=1,
        truncation_cutoff=None,
        **synthesis_kwargs,
    ):
        ws = self.f(
            z,
            None,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        img = self.g(ws, **synthesis_kwargs)
        return img

    def prep_for_train_phase(self):
        self.requires_grad_(True)
