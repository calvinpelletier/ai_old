#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models.encode.e4e import E4eEncoder
from ai_old.util.factory import build_model_from_exp
import copy
from ai_old.util.params import init_params


class E4e(nn.Module):
    def __init__(self,
        cfg,
        imsize=128,
        e_type='e4e',
        g_exp='facegen/9/0',
    ):
        super().__init__()
        self.imsize = imsize
        self.intermediate = 'modspace'

        # load original generator and config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G_ema')
        self.num_ws = og_G.num_ws
        # w_avg = og_G.f.w_avg.unsqueeze(0).unsqueeze(0)
        w_avg = og_G.f.w_avg.clone().detach()
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        del og_G

        # build encoder
        if e_type == 'e4e':
            self.e = E4eEncoder(w_avg)
        else:
            raise Exception(e_type)
        self.e.apply(init_params())

    def forward(self, x):
        ws = self.e(x, return_delta_loss=False)
        return self.g(ws)

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
