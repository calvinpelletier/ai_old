#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
from ai_old.util.factory import build_model_from_exp


class SynthSwapDeltaLoss(nn.Module):
    def __init__(self, ss_exp='wswap/1/0'):
        super().__init__()

        swapper = build_model_from_exp(ss_exp, 'G', return_cfg=False)
        self.swapper = swapper.f.to('cuda')
        self.swapper.eval()

    def forward(self, new_w, w, gender):
        assert gender.shape == (gender.shape[0], 1)
        ss_w = self.swapper(w, gender, magnitude=1.)[0]
        return F.mse_loss(new_w, ss_w)
