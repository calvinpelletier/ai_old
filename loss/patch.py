#!/usr/bin/env python3
import torch
import torch.nn as nn

# loss functions for multi-res patch discriminators


class PatchHingeDLoss(nn.Module):
    def forward(self, fake_preds, real_preds):
        assert isinstance(fake_preds, list)
        assert isinstance(real_preds, list)
        assert len(fake_preds) == len(real_preds)
        loss = 0
        # iterate over discriminators
        for fake, real in zip(fake_preds, real_preds):
            loss += self._loss(fake, real)
        return loss / len(fake_preds)

    def _loss(self, fake, real):
        fake_loss = -torch.mean(torch.min(-fake - 1, fake * 0))
        real_loss = -torch.mean(torch.min(real - 1, real * 0))
        return fake_loss + real_loss


class PatchHingeGLoss(nn.Module):
    def forward(self, fake_preds):
        assert isinstance(fake_preds, list)
        loss = 0
        for fake in fake_preds:
            loss += self._loss(fake)
        return loss / len(fake_preds)

    def _loss(self, fake):
        return -torch.mean(fake)


class PatchFeatMatchLoss(nn.Module):
    def __init__(self, type='l1'):
        super().__init__()
        assert type in ['l1', 'l2']
        self.loss = nn.L1Loss() if type == 'l1' else nn.MSELoss()

    def forward(self, fake, real):
        assert isinstance(fake, list)
        assert isinstance(real, list)
        num_d = len(fake)
        weight = 1.0 / num_d
        ret = torch.cuda.FloatTensor(1).fill_(0)
        for i in range(num_d):
            for j in range(len(fake[i])):
                loss = self.loss(fake[i][j], real[i][j].detach())
                ret += weight * loss
        return ret.squeeze()
