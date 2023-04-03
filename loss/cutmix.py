#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
import numpy as np


class CutMixReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.consistency_loss_weight = 0.2

    def forward(self, d_model, d_real_dec, d_fake_dec, g_fake, y):
        _, _, h, w = d_real_dec.shape
        mask = _cutmix_mask(
            torch.ones_like(d_real_dec),
            torch.zeros_like(d_real_dec),
            _cutmix_coordinates(h, w)
        )

        if random() > 0.5:
            mask = 1 - mask

        d_in = _cutmix(y, g_fake, mask)
        d_out = d_model(d_in, 'cutmix')

        enc_loss = F.relu(1 - d_out['d_cutmix_enc']).mean()
        dec_loss =  F.relu(1 + (mask * 2 - 1) * d_out['d_cutmix_dec']).mean()
        main_loss = enc_loss + dec_loss

        target = _cutmix(d_real_dec, d_fake_dec, mask)
        consistency_loss = F.mse_loss(d_out['d_cutmix_dec'], target)

        return main_loss + consistency_loss * self.consistency_loss_weight


def _cutmix_mask(src, target, coords):
    src, target = map(torch.clone, (src, target))
    ((y0, y1), (x0, x1)), _ = coords
    src[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return src


def _cutmix(src, target, mask):
    return src * mask + (1 - mask) * target


def _cutmix_coordinates(height, width):
    lam = np.random.beta(1., 1.)
    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))
    return ((y0, y1), (x0, x1)), lam
