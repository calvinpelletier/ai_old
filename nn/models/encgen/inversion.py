#!/usr/bin/env python3
import torch.nn as nn
from external.sg2 import persistence
from external.sg2.unit import SynthesisNetwork
from ai_old.nn.models.encode.squeeze import ZEncoder
from ai_old.util import config
import copy
from ai_old.util.params import init_params
from ai_old.util.factory import build_model_from_exp


@persistence.persistent_class
class GanInversionUnit(nn.Module):
    def __init__(self,
        cfg,
        g_exp='facegen/8/0',
        e_type='slow_squeeze_excite',
        e_nc_base=32,
        e_n_layers_per_res=[2, 4, 8, 4, 2],
        e_norm='batch',
        e_weight_norm=False,
        e_actv='mish',
    ):
        super().__init__()

        # load the original generator and its config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G_ema')

        # transfer modules from the original generator
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        self.f = copy.deepcopy(og_G.f)
        self.f.requires_grad_(False)
        del og_G

        self.z_dims = og_cfg.model.G.z_dims
        self.imsize = og_cfg.model.G.imsize

        # build encoder
        assert e_type == 'slow_squeeze_excite'
        self.e = ZEncoder(
            input_imsize=self.imsize,
            smallest_imsize=4,
            z_dims=self.z_dims,
            nc_in=3,
            nc_base=e_nc_base,
            n_layers_per_res=e_n_layers_per_res,
            norm=e_norm,
            weight_norm=e_weight_norm,
            actv=e_actv,
        )
        self.e.apply(init_params())

    def forward(self, z):
        ws = self.f(z, None)
        gen_img = self.g(ws)
        enc_w = self.e(gen_img)
        rec_img = self.g(enc_w.unsqueeze(1).repeat([1, self.f.num_ws, 1]))
        return gen_img, rec_img

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
