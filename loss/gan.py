#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from ai_old.loss.patch import PatchHingeDLoss, PatchHingeGLoss
from ai_old.loss.base import BaseLoss


def get_gan_loss_for_d(type):
    if type == 'patch_hinge':
        return PatchHingeDLoss()
    elif type == 'softplus':
        raise Exception('see TODO at class def')
        return SoftplusDLoss()
    elif type == 'hinge':
        return HingeDLoss()
    else:
        raise ValueError('unknown gan loss type: ' + type)


def get_gan_loss_for_g(type):
    if type == 'patch_hinge':
        return PatchHingeGLoss()
    elif type == 'softplus':
        raise Exception('see TODO at class def')
        return SoftplusGLoss()
    elif type == 'hinge':
        return HingeGLoss()
    else:
        raise ValueError('unknown gan loss type: ' + type)


class GLoss(BaseLoss):
    def __init__(self, conf):
        super().__init__()
        self.loss = get_gan_loss_for_g(conf.gan.type)

    def forward(self, ents, batch_num=None):
        loss = self.loss(ents['d_fake'])
        return loss, None, None # loss, sublosses, subloss_times


class DLoss(BaseLoss):
    def __init__(self, conf):
        super().__init__()
        self.loss = get_gan_loss_for_d(conf.gan.type)

    def forward(self, ents, batch_num=None):
        loss = self.loss(ents['d_fake'], ents['d_real'])
        return loss, None, None # loss, sublosses, subloss_times


class HingeDLoss(nn.Module):
    def forward(self, fake, real):
        # fake_loss = -torch.mean(torch.min(-fake - 1, fake * 0))
        # real_loss = -torch.mean(torch.min(real - 1, real * 0))
        # return fake_loss + real_loss
        return torch.mean(F.relu(1 + real) + F.relu(1 - fake))


class HingeGLoss(nn.Module):
    def forward(self, fake):
        return torch.mean(fake)


# TODO: need to verify that these softplus losses work with current
# implementation of gradient penalty

class SoftplusGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, fake):
        return self.softplus(-fake).mean()


class SoftplusDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, fake, real):
        loss_fake = self.softplus(fake).mean()
        loss_real = self.softplus(-real).mean()
        return loss_fake + loss_real
