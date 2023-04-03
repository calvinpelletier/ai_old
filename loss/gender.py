#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.util.factory import build_model_from_exp
import copy
from ai_old.util.etc import resize_imgs
import external.sg2.misc as misc


class GenderLoss(nn.Module):
    def __init__(self, l2=False, exp='gender-pred/1/2'):
        super().__init__()
        self.l2 = l2
        model = build_model_from_exp(exp, 'C', return_cfg=False)
        self.imsize = model.imsize
        self.e = copy.deepcopy(model.e)
        self.e.requires_grad_(False)
        self.c = copy.deepcopy(model.c)
        self.c.requires_grad_(False)
        del model

    def forward(self, img, target_gender, avg_batch=True):
        misc.assert_shape(target_gender, [None, 1])
        img = resize_imgs(img, self.imsize)
        enc = self.e(img)
        pred = self.c(enc)
        pred = torch.sigmoid(pred)
        if self.l2:
            loss = F.mse_loss(
                pred,
                target_gender.squeeze(),
                reduction='mean' if avg_batch else 'none',
            )
        else:
            loss = torch.abs(pred - target_gender.squeeze())
            if avg_batch:
                loss = loss.mean()
        return loss
