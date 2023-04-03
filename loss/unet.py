#!/usr/bin/env python3
import torch.nn as nn
import torch.nn.functional as F
from ai_old.loss import ComboLoss
from ai_old.loss.gan import get_gan_loss_for_d
from ai_old.loss.reg import gradient_penalty
from ai_old.loss.perceptual.lpips import LpipsLoss
from ai_old.loss.cutmix import CutMixReg


# TODO: apply gradient penalty to dec as well (and use its current weight)
class UnetDLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)

        if hasattr(conf, 'real_label'):
            real_label = conf.real_label
        else:
            real_label = 'y'
        fake_label = 'g_fake' # fake label is always g_fake (see cutmix trainer)

        self.create_subloss(
            'enc',
            DEncDecLoss(),
            ('d_fake_enc', 'd_real_enc'),
        )
        self.create_subloss(
            'dec',
            DEncDecLoss(),
            ('d_fake_dec', 'd_real_dec'),
        )

        if hasattr(conf, 'gp_enc'):
            self.create_subloss(
                'gp_enc',
                gradient_penalty,
                ('d_real_enc', real_label),
                requires_grad_for_y=True,
            )

        if hasattr(conf, 'gp_dec'):
            self.create_subloss(
                'gp_dec',
                gradient_penalty,
                ('d_real_dec', real_label),
                requires_grad_for_y=True,
            )

        if hasattr(conf, 'cutmix'):
            self.create_subloss(
                'cutmix',
                CutMixReg(),
                (
                    'd_model',
                    'd_real_dec',
                    'd_fake_dec',
                    fake_label,
                    real_label,
                ),
            )


class UnetGLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)
        self.create_subloss(
            'enc',
            GEncLoss(),
            ('d_fake_enc',),
        )
        self.create_subloss(
            'dec',
            GDecLoss(),
            ('d_fake_dec',),
        )

        if hasattr(conf, 'lpips'):
            self.create_subloss(
                'lpips',
                LpipsLoss(),
                ('g_fake', 'y'),
            )


class DEncDecLoss(nn.Module):
    def forward(self, fake, real):
        return (F.relu(1 + real) + F.relu(1 - fake)).mean()


class GEncLoss(nn.Module):
    def forward(self, fake):
        return fake.mean()


class GDecLoss(nn.Module):
    def forward(self, fake):
        return F.relu(1 + fake).mean()
