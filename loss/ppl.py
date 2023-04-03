#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.loss import ComboLoss
from ai_old.loss.gan import get_gan_loss_for_g
from ai_old.util.ema import EMA
import math
import numpy as np


class PplRegGLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)

        # gan loss
        self.create_subloss(
            'gan',
            get_gan_loss_for_g(conf.gan.type),
            ('d_fake',),
        )

        # ppl reg
        self.create_subloss(
            'ppl',
            _PplReg(),
            ('g_fake', 'zz'),
        )


class _PplReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.pl_mean = 0.
        self.pl_length_ma = EMA(0.99)

    def forward(self, g_fake, zz):
        if not zz.requires_grad:
            return torch.tensor(0., device=zz.device)

        pl_lengths = _calc_path_lengths(g_fake, zz)
        avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

        if self.pl_mean is None:
            self.pl_mean = avg_pl_length
        else:
            self.pl_mean = self.pl_length_ma.update_average(
                self.pl_mean,
                avg_pl_length,
            )

        return ((pl_lengths - self.pl_mean) ** 2).mean()


def _calc_path_lengths(g_fake, zz):
    device = g_fake.device
    num_pixels = g_fake.shape[2] * g_fake.shape[3]
    pl_noise = torch.randn(g_fake.shape, device=device) / math.sqrt(num_pixels)
    outputs = (g_fake * pl_noise).sum()

    pl_grads = torch.autograd.grad(
        outputs=outputs,
        inputs=zz,
        grad_outputs=torch.ones(outputs.shape, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()
