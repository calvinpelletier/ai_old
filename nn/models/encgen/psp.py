#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models.encode.psp_256 import Psp256Encoder
from ai_old.nn.models.encode.psp import PspEncoder, RestylePspEncoder
from ai_old.util.factory import build_model_from_exp
import copy
from ai_old.util.params import init_params


class Psp(nn.Module):
    def __init__(self,
        cfg,
        imsize=128,
        e_type='psp',
        g_exp='facegen/9/0',
    ):
        super().__init__()
        self.imsize = imsize
        self.intermediate = 'modspace'

        # load original generator and config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G_ema')
        self.num_ws = og_G.num_ws
        w_avg = og_G.f.w_avg.clone().detach()
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        del og_G

        # build encoder
        if e_type == 'psp_256':
            assert imsize == 256
            self.e = Psp256Encoder(w_avg)
        elif e_type == 'psp':
            self.e = PspEncoder(w_avg)
        self.e.apply(init_params())

    def forward(self, x):
        ws = self.e(x)
        return self.g(ws)

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)


class RestylePsp(nn.Module):
    def __init__(self,
        cfg,
        imsize=128,
        e_type='psp',
        g_exp='facegen/9/0',
    ):
        super().__init__()
        self.imsize = imsize
        self.intermediate = 'modspace'
        self.n_iters = 5

        # load original generator and config
        og_G, og_cfg = build_model_from_exp(g_exp, 'G_ema')
        self.num_ws = og_G.num_ws
        avg_ws = og_G.f.w_avg.unsqueeze(0).unsqueeze(0).repeat(1, self.num_ws, 1)
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        del og_G

        # build encoder
        if e_type == 'psp_256':
            assert imsize == 256
            raise Exception('todo')
            # self.e = Psp256Encoder(w_avg)
        elif e_type == 'psp':
            self.e = RestylePspEncoder(avg_ws)
        self.e.apply(init_params())

    def calc_avg_img(self):
        avg_img = self.g(self.e.avg_ws)
        self.e.set_avg_img(avg_img)

    def forward(self, input_img):
        img = None
        ws = None
        for i in range(self.n_iters):
            ws = self.e(input_img, img, ws)
            img = self.g(ws)
        return img

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
